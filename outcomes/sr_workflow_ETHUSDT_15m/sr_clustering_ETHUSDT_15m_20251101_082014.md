# SR Workflow Summary Report

**Generated:** 2025-11-01 08:20:59
**Symbol:** ETHUSDT
**Exchange:** binance
**Timeframe:** 15m
**Direction:** long
**Mode:** light

---

## Workflow Execution Summary

- **Total Duration:** 44.85 seconds
- **Steps Completed:** 3/3
- **Steps Failed:** 0/3
- **Success Rate:** 100.0%
- **Start Time:** N/A
- **End Time:** 2025-11-01 08:20:59.319707

## Steps Completed

✅ sr_parameter_optimization
✅ sr_detection
✅ sr_clustering

## Artifacts Created

### optimization

- **sr_parameter_optimization_result:** `artifacts/pre_training/long/Analyst/sr_parameter_optimization/sr_parameter_optimization_sr_parameter_optimization_result_long_Analyst_20251101_082013.parquet`

### detection

- **sr_detection_result:** `outcomes/sr_workflow_ETHUSDT_15m/sr_detection_ETHUSDT_15m_20251101_082014.json`

### clustering

- `artifacts/pre_training/ETHUSDT/binance/long/Analyst/sr_clustering/sr_clustering_sr_clustering_result_ETHUSDT_binance_long_Analyst_20251101_082059.parquet`
- `artifacts/pre_training/ETHUSDT/binance/long/Analyst/sr_clustering/sr_clustering_sr_levels_dictionary_ETHUSDT_binance_long_Analyst_20251101_082059.parquet`

## Metrics Summary

```json
{
  "optimization": {
    "data_points": 105092,
    "optimization_time": 32.56007719039917,
    "best_score": 0.9565217391304348,
    "total_combinations_tested": 12,
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
    "total_levels": 85,
    "support_levels": 41,
    "resistance_levels": 44,
    "ml_model_used": true
  },
  "clustering": {
    "total_clusters": 2,
    "clustering_efficiency": 0.5,
    "execution_mode": "light",
    "enhancement_features": {
      "hardware_optimization": true,
      "vectorbt_optimization": true,
      "memory_optimization": true,
      "gpu_acceleration": true
    },
    "performance_metrics": {
      "clustering_time": 0.148106,
      "levels_per_second": 27.007683686008672,
      "clusters_per_second": 13.503841843004336,
      "memory_usage_mb": 50.0,
      "cpu_utilization": 0.4,
      "gpu_utilization": 0.1,
      "optimization_gains": {
        "vectorbt_speedup": 2.0,
        "hardware_optimization": 1.3,
        "memory_optimization": 1.2,
        "total_gain": 1.0
      }
    },
    "quality_metrics": {
      "clustering_coverage": 1.0,
      "average_cluster_size": 2.0,
      "cluster_size_std": 0.0,
      "total_clusters": 2,
      "reduction_ratio": 0.5,
      "silhouette_score": -0.21719181004030905,
      "calinski_harabasz_score": 0.4844903988183123,
      "davies_bouldin_score": 2.0317600203979818,
      "quality_score": -0.41615578001315745,
      "high_quality_clusters": 0,
      "quality_ratio": 0.0,
      "meets_quality_threshold": false
    },
    "hardware_metrics": {
      "hardware_optimization_enabled": true,
      "gpu_acceleration_enabled": true,
      "memory_optimization_enabled": true,
      "batch_processing_enabled": true,
      "batch_size": 1000,
      "memory_limit_gb": 8.0,
      "cpu_cores": 4,
      "gpu_available": false,
      "gpu_type": null,
      "memory_gb": 8.0
    },
    "basestep_integration": {
      "integration_valid": true,
      "artifacts_saved": 2,
      "required_artifacts": [
        "sr_clustering_result",
        "sr_levels_dictionary"
      ],
      "step_name": "sr_clustering"
    }
  }
}
```

## Individual Step Reports

- [sr_parameter_optimization](outcomes/sr_workflow_ETHUSDT_15m/sr_parameter_optimization_ETHUSDT_15m_20251101_082014.md)
- [sr_detection](outcomes/sr_workflow_ETHUSDT_15m/sr_detection_ETHUSDT_15m_20251101_082014.md)
- [sr_clustering](outcomes/sr_workflow_ETHUSDT_15m/sr_clustering_ETHUSDT_15m_20251101_082014.md)
