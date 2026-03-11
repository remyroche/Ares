# Backtest Protocol

This document defines the **economic evaluation framework** for trading models.

Backtests must simulate realistic trading conditions.

---

# 1. Signal Timing

Signals generated at time t must be executed at **t+1 or later**.

This prevents lookahead bias.

---

# 2. Transaction Costs

Backtests must include transaction costs. Assume 0.3% per round trad

Minimum assumptions:

commission  
spread  
market impact (if applicable)

Cost parameters must be documented.

---

# 3. Slippage

Assume none

---

# 4. Turnover Constraints

Strategies must report turnover.

High turnover signals may be economically infeasible.

---

# 5. Liquidity Filters

None: prefer limit orders

---

# 6. Position Sizing

Position sizing rules must be explicit.

Examples:

equal weight  
volatility targeting  
risk parity  
signal proportional

---

# 7. Portfolio Construction

If signals are cross-sectional, portfolio construction rules must be defined.

Examples:

long-short  
top/bottom quantiles  
rank weighting

---

# 8. Evaluation Metrics

Required metrics:

PnL/day
trades/day
annualized return  
Sortino ratio  
maximum drawdown  
turnover  

Additional metrics may include:

Sharpe  ratio  
Calmar ratio  
hit rate
Ulcer
Time under water

---

# 9. Out-of-Sample Evaluation

Backtests must report **in-sample and out-of-sample results separately**.

Only out-of-sample performance should be considered predictive.
