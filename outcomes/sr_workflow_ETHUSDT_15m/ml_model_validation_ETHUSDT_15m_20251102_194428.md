# ML MODEL VALIDATION Report

**Generated:** 2025-11-02 19:44:49
**Symbol:** ETHUSDT
**Exchange:** binance
**Timeframe:** 15m
**Direction:** long
**Mode:** light

---

## Execution Summary

- **Status:** ✅ Success
- **Duration:** 0.01 seconds
- **Step:** ml_model_validation

## Metrics

```json
{
  "tests_passed": 4,
  "total_tests": 6,
  "success_rate": 66.66666666666666,
  "precision_at_10": 1.0,
  "spearman_rho": NaN,
  "separation": 1.1102230246251565e-16
}
```

## Artifacts Created

- **ml_validation:** {'validation_results': {'precision_at_k': {5: 1.0, 10: 1.0, 20: 1.0, 50: 1.0}, 'spearman': nan, 'spearman_pvalue': nan, 'separation': {'mean_strong': 0.552714777496176, 'mean_weak': 0.5527147774961759, 'median_strong': 0.552714777496176, 'median_weak': 0.552714777496176, 'separation': 1.1102230246251565e-16, 'weak_above_strong_median_pct': 100.0, 'strong_below_weak_median_pct': 100.0}, 'future_generalization': {'r2': None}, 'sample_size_check': {'total_samples': 317, 'strong_samples': 50}}, 'tests_passed': 4, 'total_tests': 6, 'success_rate': 66.66666666666666}
