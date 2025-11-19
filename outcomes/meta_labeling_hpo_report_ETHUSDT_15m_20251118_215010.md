# Meta-Labeling HPO Report

**Generated:** 2025-11-18 21:50:10 UTC

**Symbol:** ETHUSDT | **Exchange:** binance | **Timeframe:** 15m

---

## Summary

- **Total Configurations Evaluated:** 28
- **Pareto Frontier Size:** 2
- **Best Combined Score:** 0.218165
- **Optimization Method:** Bayesian TPE with Pareto Frontier

## Best Parameters (Highest Combined Score)

```json
{
  "profit_thr_base": 0.012392823207483914,
  "stop_to_profit_ratio": 0.49026674221303634,
  "horizon_bars": 20,
  "min_event_spacing": 2,
  "iso_min_prob": 0.06075448519014384,
  "target_clip_high_q": 0.9153471711318563,
  "kalman_Q": 1.3492834268013232e-05,
  "kalman_R": 0.07902619549708234,
  "vol_baseline_window": 188,
  "profit_mult_min": 0.9041986740582306,
  "profit_mult_max": 1.3046137691733706,
  "stop_mult_min": 0.5488360570031919,
  "stop_mult_max": 1.6842330265121568
}
```

## Recommended Parameters (Pareto Knee Point)

Balanced trade-off between learnability and profitability:

- **Learnability:** 0.0690
- **Profitability:** 56.6249
- **Mean AUC:** 0.5958
- **Sharpe (Winners):** 5.5507

```json
{
  "profit_thr_base": 0.012392823207483914,
  "stop_to_profit_ratio": 0.49026674221303634,
  "horizon_bars": 20,
  "min_event_spacing": 2,
  "iso_min_prob": 0.06075448519014384,
  "target_clip_high_q": 0.9153471711318563,
  "kalman_Q": 1.3492834268013232e-05,
  "kalman_R": 0.07902619549708234,
  "vol_baseline_window": 188,
  "profit_mult_min": 0.9041986740582306,
  "profit_mult_max": 1.3046137691733706,
  "stop_mult_min": 0.5488360570031919,
  "stop_mult_max": 1.6842330265121568
}
```

## Pareto Frontier Analysis

The Pareto frontier contains 2 non-dominated solutions representing optimal trade-offs between learnability and profitability.

### Top 10 Pareto Solutions

| Rank | Learnability | Profitability | Combined | Mean AUC | Sharpe | N Events |
|------|-------------|--------------|----------|----------|--------|----------|
| 1 | -0.1835 | 6499998.1166 | 19499.8659 | 0.5566 | 649999.9970 | 450 |
| 2 | 0.0690 | 56.6249 | 0.2182 | 0.5958 | 5.5507 | 431 |

## Regularization Checks

All configurations were evaluated with:

1. **Temporal Stability:** Rolling window AUC variance penalty
2. **Learnability Threshold:** Mean AUC < 0.7 heavily penalized
3. **Profit/Stop Constraint:** Profit threshold must be ≥ 1.5× stop threshold
4. **Label Balance:** Entropy-based balance scoring

## Artifacts

- **Best Params JSON:** `meta_labeling_hpo_best_params_ETHUSDT_15m_20251118_215010.json`
- **Candidate Pool CSV:** `meta_labeling_hpo_candidate_pool_ETHUSDT_15m_20251118_215010.csv`
- **Pareto Frontier CSV:** `meta_labeling_hpo_pareto_front_ETHUSDT_15m_20251118_215010.csv`
- **This Report:** `meta_labeling_hpo_report_ETHUSDT_15m_20251118_215010.md`

