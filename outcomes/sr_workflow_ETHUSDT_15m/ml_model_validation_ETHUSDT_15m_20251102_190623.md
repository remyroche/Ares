# ML MODEL VALIDATION Report

**Generated:** 2025-11-02 19:07:11
**Symbol:** ETHUSDT
**Exchange:** binance
**Timeframe:** 15m
**Direction:** long
**Mode:** light

---

## Execution Summary

- **Status:** ✅ Success
- **Duration:** 0.03 seconds
- **Step:** ml_model_validation

## Metrics

```json
{
  "tests_passed": 5,
  "total_tests": 6,
  "success_rate": 83.33333333333334,
  "precision_at_10": 1.0,
  "spearman_rho": 0.8522441826579998,
  "separation": 0.23785273612472102
}
```

## Artifacts Created

- **ml_validation:** {'validation_results': {'precision_at_k': {5: 1.0, 10: 1.0, 20: 1.0, 50: 1.0}, 'spearman': 0.8522441826579998, 'spearman_pvalue': 6.169953778223678e-24, 'separation': {'mean_strong': 0.7860932250173134, 'mean_weak': 0.5482404888925924, 'median_strong': 0.7720786777563549, 'median_weak': 0.5473239770935154, 'separation': 0.23785273612472102, 'weak_above_strong_median_pct': 0.0, 'strong_below_weak_median_pct': 0.0}, 'future_generalization': {'r2': None}, 'sample_size_check': {'total_samples': 319, 'strong_samples': 81}}, 'tests_passed': 5, 'total_tests': 6, 'success_rate': 83.33333333333334}
