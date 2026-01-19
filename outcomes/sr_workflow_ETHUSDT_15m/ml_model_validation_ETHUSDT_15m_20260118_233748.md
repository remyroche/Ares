# ML MODEL VALIDATION Report

**Generated:** 2026-01-18 23:40:16
**Symbol:** ETHUSDT
**Exchange:** binance
**Timeframe:** 15m
**Direction:** long
**Mode:** light

---

## Execution Summary

- **Status:** ✅ Success
- **Duration:** 0.45 seconds
- **Step:** ml_model_validation

## Metrics

```json
{
  "tests_passed": 4,
  "total_tests": 7,
  "success_rate": 57.14285714285714,
  "precision_at_10": 1.0,
  "spearman_rho": 0.044740894051612336,
  "separation": 0.0018223851645247091
}
```

## Artifacts Created

- **ml_validation:** {'validation_results': {'precision_at_k': {5: 1.0, 10: 1.0, 20: 1.0, 50: 1.0}, 'spearman': 0.044740894051612336, 'spearman_pvalue': 0.356384285983798, 'separation': {'mean_strong': 0.520997707179331, 'mean_weak': 0.5191753220148063, 'median_strong': 0.5215111776889153, 'median_weak': 0.5215111776889153, 'separation': 0.0018223851645247091, 'weak_above_strong_median_pct': 70.16949152542374, 'strong_below_weak_median_pct': 100.0}, 'future_generalization': {'r2': -12.38754306870877, 'train_period': '2024-01-21 12:00:00 to 2024-03-08 12:00:00', 'test_period': '2024-03-09 00:00:00 to 2024-03-20 12:00:00', 'train_strong_count': 269, 'test_strong_count': 158}, 'sample_size_check': {'total_samples': 1691, 'strong_samples': 427}}, 'tests_passed': 4, 'total_tests': 7, 'success_rate': 57.14285714285714}
