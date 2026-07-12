# Backtest And Policy Evaluation Protocol

Backtests must evaluate executable decisions under the same score, cost, exit,
and portfolio contracts used downstream.

## 1. Decision And Execution Timing

- A feature timestamp is the time the row becomes observable.
- A decision may use only completed inputs at that timestamp.
- Entry must occur at the first explicitly executable price after the decision.
- Same-bar high/low information may define a future outcome, never an entry.
- Multi-frequency joins must use backward/as-of alignment and recorded latency.

## 2. Costs

Record costs by layer and unit. Do not use one undocumented global constant.

- Label cost: cost embedded in the supervised economic target, if any.
- Policy stress cost: the conservative round-trip assumption used by replay.
  The current `simple_policy_optimiser` default is `0.01`, meaning 1% of
  notional per round trip.
- Live execution cost: realized fees, spread, slippage, and fill shortfall.

Reports must state whether a value is decimal return, percent return, bps, or
currency. Reconcile embedded and replay costs so fees are not counted twice.

## 3. Entry And Exit Simulation

- Apply side-correct TP, SL, trailing activation, trailing gap, and timeout.
- A timeout exits at the last executable price with costs; it is not a full loss.
- Stops use the configured executable stop semantics and may include gap/slip.
- Report full-stop, timeout, holding time, MFE, MAE, and exit-cause distributions.
- Geometry may vary by side and archetype. Global fallback geometry must be
  recorded when local support is insufficient.

## 4. Liquidity And Capacity

- Include point-in-time spread and liquidity checks when available.
- Prefer limit-order assumptions only when the fill model supports them.
- Report rejected/unfilled rows and do not silently treat them as fills.
- Enforce concurrent-position, wallet exposure, and capital-pressure limits.
- Portfolio ranking must use the same calibrated EV used for admission and size.

## 5. OOS Policy Optimization

- Optimize on training folds and score only each fold's non-training rows.
- For fold scores use the documented stability objective. The current standard is
  `mean - 0.5 * std + 0.25 * worst` unless a manifest explicitly overrides it.
- Side-by-archetype estimates must shrink toward the side parent according to
  support and fold stability.
- Thresholds, calibrators, hit-rate surprise, and portfolio parameters must be
  fixed before evaluating the next OOS window.

## 6. Required Metrics

Report overall, month, week, side, and archetype slices:

- selected trades and trades/day
- top-5/10/20/30% mean and sum net EV where applicable
- notional net return per trade
- bankroll/portfolio PnL after sizing
- hit rate and positive-net rate
- full-stop and timeout rates
- turnover, exposure, concentration, drawdown, and worst fold/week/month
- gross, fees, spread, slippage, and net reconciliation
- signed residual mean/autocorrelation and signed hit-rate surprise, with support

Only leakage-safe OOS or frozen replay evidence supports promotion.
