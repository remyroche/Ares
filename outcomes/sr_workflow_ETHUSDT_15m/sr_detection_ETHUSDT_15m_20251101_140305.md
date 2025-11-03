# SR Workflow Summary Report

**Generated:** 2025-11-01 14:04:00
**Symbol:** ETHUSDT
**Exchange:** binance
**Timeframe:** 15m
**Direction:** long
**Mode:** light

---

## Workflow Execution Summary

- **Total Duration:** 54.81 seconds
- **Steps Completed:** 3/3
- **Steps Failed:** 0/3
- **Success Rate:** 100.0%
- **Start Time:** N/A
- **End Time:** 2025-11-01 14:04:00.485464

## Steps Completed

✅ sr_parameter_optimization
✅ sr_detection
✅ sr_filtering

## Artifacts Created

### optimization

- **sr_parameter_optimization_result:** `artifacts/pre_training/long/Analyst/sr_parameter_optimization/sr_parameter_optimization_sr_parameter_optimization_result_long_Analyst_20251101_140305.parquet`

### detection

- **sr_detection_result:** `outcomes/sr_workflow_ETHUSDT_15m/sr_detection_ETHUSDT_15m_20251101_140305.json`

### filtering

- **filtered_sr_levels:** `151`
- **removed_weak_levels:** `9`
- **strength_threshold:** `0.5`

## Metrics Summary

```json
{
  "optimization": {
    "data_points": 105092,
    "optimization_time": 37.83482098579407,
    "best_score": 0.9487210233079597,
    "total_combinations_tested": 76891,
    "performance_improvements": {
      "vectorbt_speedup": 1.0,
      "hardware_optimization_gains": {
        "cpu_optimization": 1.0,
        "memory_optimization": 1.0,
        "gpu_acceleration": 1.0
      },
      "bayesian_efficiency": 0.0
    }
  },
  "detection": {
    "total_levels": 160,
    "support_levels": 94,
    "resistance_levels": 66,
    "ml_model_used": true
  },
  "filtering": {
    "total_levels_before": 160,
    "total_levels_after": 151,
    "weak_levels_removed": 9,
    "retention_rate": 0.94375
  }
}
```

## Individual Step Reports

- [sr_parameter_optimization](outcomes/sr_workflow_ETHUSDT_15m/sr_parameter_optimization_ETHUSDT_15m_20251101_140305.md)
- [sr_detection](outcomes/sr_workflow_ETHUSDT_15m/sr_detection_ETHUSDT_15m_20251101_140305.md)
