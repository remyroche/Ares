# Iterative Optimization Hyperparameter Tuning Report
**Generated**: 2025-10-30 21:24:55
**Dataset**: 480 samples, 14 features

## Optimization Summary

**Total Trials**: 20
**Best Composite Score**: -10.0000

### Best Configuration Metrics
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| CV Score | 0.9368 | ≥1.2 | ⚠️ |
| Silhouette Score | -0.0318 | ≥0.1 | ⚠️ |
| DBI Score | 17.3810 | ≤2.0 | ⚠️ |
| Balance Score | 0.9419 | ≥0.3 | ✅ |
| Temporal Smoothness | 0.1608 | ≥0.6 | ⚠️ |
| Number of Clusters | 6 | 4-6 | ✅ |
| Cluster Sizes Valid | True | 2%-20% | ✅ |

### Complete Best Parameters
```json
{
  "K_MIN": 4,
  "K_MAX": 5,
  "max_rounds": 42,
  "local_churn_cap": 5395,
  "knn_size": 18,
  "w_cv": 0.5467983561008608,
  "w_sil": 0.05871254182522992,
  "w_temp": 0.273235229154987,
  "w_bal": 0.06808920093945671,
  "MIN_FRAC": 0.04124217733388137,
  "MAX_FRAC": 0.15205844942958024,
  "eps_std_step1": -0.10601802956760115,
  "sil_guard": -0.058377867959978916,
  "temporal_bonus": 0.19246782213565522,
  "eps_cv": 2.310201887845294e-06,
  "eps_sil": 2.3270677083837777e-05,
  "eps_temp": 4.059611610484306e-05,
  "size_gate_base": 0.0001673888154257967,
  "size_gate_alpha": 0.02727780074568463,
  "size_gate_beta": 0.037473748411882515
}
```
