# Meta-Labeling HPO Report

**Generated:** 2025-11-18 21:37:24 UTC

**Symbol:** ETHUSDT | **Exchange:** binance | **Timeframe:** 15m

---

## Summary

- **Total Configurations Evaluated:** 0
- **Pareto Frontier Size:** 0
- **Best Combined Score:** -1000000000.000000
- **Optimization Method:** Bayesian TPE with Pareto Frontier

## Best Parameters (Highest Combined Score)

```json
{
  "profit_thr_base": 0.016239882614641973,
  "stop_to_profit_ratio": 0.651764293371669,
  "horizon_bars": 23,
  "min_event_spacing": 6,
  "iso_min_prob": 0.015601864044243652,
  "target_clip_high_q": 0.9140395068302583,
  "kalman_Q": 1.3066739238053272e-05,
  "kalman_R": 0.05399484409787434,
  "vol_baseline_window": 135,
  "profit_mult_min": 0.8540362888980227,
  "profit_mult_max": 1.0205844942958024,
  "stop_mult_min": 0.9849549260809971,
  "stop_mult_max": 1.8324426408004217
}
```

## Pareto Frontier Analysis

The Pareto frontier contains 0 non-dominated solutions representing optimal trade-offs between learnability and profitability.

### Top 10 Pareto Solutions

| Rank | Learnability | Profitability | Combined | Mean AUC | Sharpe | N Events |
|------|-------------|--------------|----------|----------|--------|----------|

## Regularization Checks

All configurations were evaluated with:

1. **Temporal Stability:** Rolling window AUC variance penalty
2. **Learnability Threshold:** Mean AUC < 0.7 heavily penalized
3. **Profit/Stop Constraint:** Profit threshold must be ≥ 1.5× stop threshold
4. **Label Balance:** Entropy-based balance scoring

## Artifacts

- **Best Params JSON:** `meta_labeling_hpo_best_params_ETHUSDT_15m_20251118_213724.json`
- **Candidate Pool CSV:** `N/A`
- **Pareto Frontier CSV:** `N/A`
- **This Report:** `meta_labeling_hpo_report_ETHUSDT_15m_20251118_213724.md`

