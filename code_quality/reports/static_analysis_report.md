# Static Analysis Report: Missing Imports & Undefined Names

**Generated:** 2025-10-16 07:40:51

## Summary

- **Total files analyzed:** 1530
- **Files with issues:** 1530
- **Total undefined names found:** 70729

## Top Files with Most Issues

1. **src/utils/ml_common/feature_selection.py** - 577 issues
2. **src/training/steps/pre_training/unified_data_driven_pipeline/consolidated_pipeline.py** - 529 issues
3. **src/training/steps/market_analysis/optimal_regime_clustering_backup/orchestrator.py** - 437 issues
4. **research/candle_based_features/advanced_candle_features.py** - 372 issues
5. **research/candle_ml_patterns/advanced_candle_features.py** - 372 issues
6. **src/training/steps/backtesting/final_parameters_optimization.py** - 337 issues
7. **exchanges/binance.py** - 337 issues
8. **src/trading/reporting/performance_reporter.py** - 304 issues
9. **src/monitoring/csv_export_manager.py** - 286 issues
10. **exchanges/okx.py** - 270 issues
11. **src/training/steps/market_analysis/clusters/iterative_optimization.py** - 269 issues
12. **src/training/steps/market_analysis/tas_regime/core/tas_regime_detector.py** - 267 issues
13. **src/training/steps/pre_training/sub_pipeline.py** - 262 issues
14. **src/feature_generation/utils/vectorbt_rolling_optimizer.py** - 252 issues
15. **src/training/steps/pre_training/multi_horizon_profit_labeler.py** - 252 issues
16. **src/utils/feature_output_validator.py** - 251 issues
17. **src/training/steps/model_training/tactician_ensemble_training.py** - 246 issues
18. **src/training/steps/pre_training/unified_data_driven_pipeline/consolidated_pipeline_runner.py** - 244 issues
19. **src/training/steps/model_training/__init__.py** - 244 issues
20. **src/training/steps/market_analysis/components/regime_models_training.py** - 239 issues

## Detailed File-by-File Analysis

### 1. src/utils/ml_common/feature_selection.py

**Total Issues:** 577

**Issues by Name:**

- **args** (appears 11 times):
  - Line 1123, Column 31
  - Line 1164, Column 34
  - Line 1121, Column 35
  - Line 1112, Column 54
  - Line 1177, Column 23
  - ... and 6 more occurrences

- **available** (appears 2 times):
  - Line 9478, Column 22
  - Line 9479, Column 30

- **bootstrap_result** (appears 1 times):
  - Line 3816, Column 36

- **chunk** (appears 5 times):
  - Line 9014, Column 65
  - Line 9017, Column 37
  - Line 809, Column 57
  - Line 825, Column 57
  - Line 817, Column 57

- **cluster** (appears 1 times):
  - Line 6112, Column 48

- **combination** (appears 7 times):
  - Line 3414, Column 41
  - Line 3417, Column 40
  - Line 3519, Column 45
  - Line 3498, Column 46
  - Line 3520, Column 49
  - ... and 2 more occurrences

- **cooccurrences** (appears 1 times):
  - Line 3529, Column 35

- **dataset** (appears 6 times):
  - Line 3175, Column 37
  - Line 3189, Column 27
  - Line 3193, Column 20
  - Line 3194, Column 20
  - Line 3195, Column 32
  - ... and 1 more occurrences

- **dataset_idx** (appears 2 times):
  - Line 3208, Column 35
  - Line 3189, Column 66

- **e** (appears 155 times):
  - Line 65, Column 50
  - Line 995, Column 29
  - Line 1162, Column 37
  - Line 1588, Column 33
  - Line 1730, Column 47
  - ... and 150 more occurrences

- **error** (appears 1 times):
  - Line 9468, Column 29

- **f** (appears 46 times):
  - Line 4347, Column 16
  - Line 7310, Column 42
  - Line 2191, Column 52
  - Line 5397, Column 53
  - Line 6158, Column 52
  - ... and 41 more occurrences

- **f_regression** (appears 1 times):
  - Line 6264, Column 32

- **fallback_error** (appears 2 times):
  - Line 1202, Column 58
  - Line 1205, Column 46

- **feature_group** (appears 5 times):
  - Line 5496, Column 31
  - Line 5480, Column 35
  - Line 5501, Column 33
  - Line 5502, Column 42
  - Line 5503, Column 51

- **feature_i** (appears 2 times):
  - Line 4318, Column 58
  - Line 4321, Column 66

- **feature_j** (appears 1 times):
  - Line 4321, Column 77

- **fold** (appears 5 times):
  - Line 4763, Column 35
  - Line 9428, Column 42
  - Line 5555, Column 57
  - Line 9421, Column 34
  - Line 9429, Column 54

- **fold_e** (appears 1 times):
  - Line 4749, Column 92

- **fold_features** (appears 1 times):
  - Line 6702, Column 27

- **fold_idx** (appears 5 times):
  - Line 4743, Column 36
  - Line 6642, Column 28
  - Line 6600, Column 56
  - Line 6650, Column 41
  - Line 4749, Column 55

- **fold_imp** (appears 1 times):
  - Line 5575, Column 40

- **freq** (appears 3 times):
  - Line 3845, Column 94
  - Line 3846, Column 105
  - Line 3847, Column 89

- **group** (appears 3 times):
  - Line 5465, Column 23
  - Line 5466, Column 53
  - Line 5468, Column 53

- **i** (appears 144 times):
  - Line 5656, Column 15
  - Line 5661, Column 24
  - Line 7759, Column 33
  - Line 950, Column 47
  - Line 3308, Column 27
  - ... and 139 more occurrences

- **idx** (appears 14 times):
  - Line 4466, Column 30
  - Line 4472, Column 71
  - Line 4031, Column 49
  - Line 4054, Column 35
  - Line 4547, Column 66
  - ... and 9 more occurrences

- **importance_data** (appears 1 times):
  - Line 5522, Column 38

- **inner_idx** (appears 2 times):
  - Line 2805, Column 41
  - Line 2817, Column 62

- **interaction** (appears 5 times):
  - Line 8606, Column 19
  - Line 8606, Column 64
  - Line 8607, Column 37
  - Line 8608, Column 40
  - Line 8607, Column 66

- **interactions** (appears 2 times):
  - Line 3460, Column 31
  - Line 3456, Column 47

- **issue** (appears 1 times):
  - Line 1689, Column 43

- **j** (appears 32 times):
  - Line 5665, Column 19
  - Line 4320, Column 28
  - Line 5671, Column 32
  - Line 3310, Column 57
  - Line 3521, Column 32
  - ... and 27 more occurrences

- **kwargs** (appears 10 times):
  - Line 1123, Column 39
  - Line 1165, Column 36
  - Line 909, Column 41
  - Line 920, Column 44
  - Line 921, Column 41
  - ... and 5 more occurrences

- **m** (appears 6 times):
  - Line 5134, Column 30
  - Line 5164, Column 32
  - Line 5185, Column 31
  - Line 5134, Column 52
  - Line 5164, Column 54
  - ... and 1 more occurrences

- **m1_e** (appears 1 times):
  - Line 4300, Column 92

- **max_dep** (appears 1 times):
  - Line 5799, Column 37

- **metric_type** (appears 3 times):
  - Line 6532, Column 19
  - Line 6533, Column 50
  - Line 6536, Column 39

- **n_est** (appears 1 times):
  - Line 5798, Column 40

- **neighbor** (appears 1 times):
  - Line 8684, Column 28

- **op** (appears 3 times):
  - Line 9487, Column 30
  - Line 9488, Column 26
  - Line 9490, Column 26

- **other** (appears 2 times):
  - Line 3547, Column 46
  - Line 3546, Column 63

- **other_idx** (appears 1 times):
  - Line 7772, Column 36

- **outer** (appears 2 times):
  - Line 2834, Column 42
  - Line 2876, Column 50

- **outer_idx** (appears 2 times):
  - Line 2827, Column 33
  - Line 2785, Column 53

- **r** (appears 5 times):
  - Line 6280, Column 30
  - Line 6281, Column 34
  - Line 6282, Column 36
  - Line 6285, Column 41
  - Line 6288, Column 36

- **regime_data** (appears 4 times):
  - Line 7211, Column 20
  - Line 7202, Column 23
  - Line 7203, Column 33
  - Line 7204, Column 33

- **s** (appears 12 times):
  - Line 2523, Column 22
  - Line 2655, Column 22
  - Line 2523, Column 75
  - Line 2655, Column 75
  - Line 2523, Column 56
  - ... and 7 more occurrences

- **selected** (appears 1 times):
  - Line 8931, Column 97

- **selected_feature** (appears 5 times):
  - Line 8023, Column 55
  - Line 8067, Column 91
  - Line 8068, Column 47
  - Line 8088, Column 91
  - Line 8089, Column 47

- **selections** (appears 2 times):
  - Line 3328, Column 32
  - Line 3328, Column 50

- **stage** (appears 2 times):
  - Line 6491, Column 52
  - Line 6491, Column 66

- **stage_data** (appears 3 times):
  - Line 2389, Column 31
  - Line 2390, Column 29
  - Line 2389, Column 57

- **stage_name** (appears 4 times):
  - Line 2392, Column 44
  - Line 2394, Column 44
  - Line 2396, Column 44
  - Line 2398, Column 44

- **start_idx** (appears 11 times):
  - Line 2991, Column 26
  - Line 6845, Column 26
  - Line 2995, Column 33
  - Line 2996, Column 33
  - Line 3004, Column 37
  - ... and 6 more occurrences

- **tool** (appears 2 times):
  - Line 1711, Column 53
  - Line 9480, Column 31

- **train_idx** (appears 6 times):
  - Line 6603, Column 43
  - Line 6605, Column 43
  - Line 4735, Column 39
  - Line 4735, Column 53
  - Line 5548, Column 48
  - ... and 1 more occurrences

- **train_test_split** (appears 1 times):
  - Line 6587, Column 55

- **v** (appears 2 times):
  - Line 6543, Column 38
  - Line 6543, Column 84

- **val_idx** (appears 4 times):
  - Line 6604, Column 41
  - Line 6606, Column 41
  - Line 5548, Column 71
  - Line 5549, Column 53

- **vbt_e** (appears 2 times):
  - Line 502, Column 79
  - Line 9304, Column 81

- **votes** (appears 1 times):
  - Line 5228, Column 23

- **w** (appears 2 times):
  - Line 2966, Column 28
  - Line 2966, Column 55

- **warning** (appears 3 times):
  - Line 9471, Column 28
  - Line 1987, Column 42
  - Line 1694, Column 43

- **word** (appears 3 times):
  - Line 6371, Column 23
  - Line 6373, Column 25
  - Line 6375, Column 25

---

### 2. src/training/steps/pre_training/unified_data_driven_pipeline/consolidated_pipeline.py

**Total Issues:** 529

**Issues by Name:**

- **ablation_config** (appears 2 times):
  - Line 6563, Column 34
  - Line 6573, Column 34

- **ablation_name** (appears 5 times):
  - Line 6562, Column 37
  - Line 6549, Column 54
  - Line 6572, Column 37
  - Line 6568, Column 51
  - Line 6571, Column 54

- **analysis** (appears 5 times):
  - Line 5736, Column 34
  - Line 5792, Column 34
  - Line 5788, Column 34
  - Line 5738, Column 35
  - Line 5795, Column 50

- **append_mode** (appears 1 times):
  - Line 6360, Column 50

- **attr_name** (appears 4 times):
  - Line 762, Column 33
  - Line 788, Column 33
  - Line 763, Column 46
  - Line 789, Column 46

- **batch_id** (appears 2 times):
  - Line 6293, Column 50
  - Line 6320, Column 66

- **col** (appears 95 times):
  - Line 6240, Column 27
  - Line 2922, Column 31
  - Line 2960, Column 33
  - Line 3006, Column 31
  - Line 4795, Column 30
  - ... and 90 more occurrences

- **column** (appears 12 times):
  - Line 5401, Column 54
  - Line 5405, Column 42
  - Line 5407, Column 37
  - Line 5411, Column 41
  - Line 5413, Column 37
  - ... and 7 more occurrences

- **display_name** (appears 6 times):
  - Line 768, Column 54
  - Line 770, Column 55
  - Line 772, Column 71
  - Line 794, Column 54
  - Line 796, Column 55
  - ... and 1 more occurrences

- **e** (appears 234 times):
  - Line 42, Column 76
  - Line 85, Column 88
  - Line 108, Column 85
  - Line 315, Column 87
  - Line 330, Column 86
  - ... and 229 more occurrences

- **end_time** (appears 1 times):
  - Line 6320, Column 56

- **exchange** (appears 6 times):
  - Line 6293, Column 30
  - Line 6320, Column 24
  - Line 6360, Column 30
  - Line 6291, Column 68
  - Line 6318, Column 68
  - ... and 1 more occurrences

- **f** (appears 9 times):
  - Line 2037, Column 37
  - Line 3468, Column 37
  - Line 5986, Column 45
  - Line 2037, Column 64
  - Line 3468, Column 74
  - ... and 4 more occurrences

- **feat** (appears 9 times):
  - Line 5681, Column 39
  - Line 5061, Column 32
  - Line 5062, Column 35
  - Line 5063, Column 42
  - Line 5064, Column 44
  - ... and 4 more occurrences

- **feature_a** (appears 2 times):
  - Line 2433, Column 50
  - Line 2434, Column 64

- **feature_b** (appears 2 times):
  - Line 2433, Column 67
  - Line 2434, Column 92

- **feature_name** (appears 8 times):
  - Line 5550, Column 25
  - Line 4084, Column 50
  - Line 4084, Column 88
  - Line 3326, Column 35
  - Line 3328, Column 110
  - ... and 3 more occurrences

- **fold_idx** (appears 4 times):
  - Line 4773, Column 33
  - Line 4786, Column 37
  - Line 4778, Column 44
  - Line 4782, Column 44

- **fs** (appears 1 times):
  - Line 3456, Column 42

- **i** (appears 25 times):
  - Line 3980, Column 38
  - Line 4018, Column 16
  - Line 3875, Column 27
  - Line 3980, Column 65
  - Line 5976, Column 40
  - ... and 20 more occurrences

- **idx** (appears 4 times):
  - Line 4645, Column 23
  - Line 4648, Column 60
  - Line 4646, Column 66
  - Line 4646, Column 103

- **interval** (appears 6 times):
  - Line 6293, Column 40
  - Line 6320, Column 34
  - Line 6360, Column 40
  - Line 6291, Column 80
  - Line 6318, Column 80
  - ... and 1 more occurrences

- **issue** (appears 5 times):
  - Line 1805, Column 35
  - Line 1813, Column 42
  - Line 2170, Column 46
  - Line 1805, Column 93
  - Line 1805, Column 121

- **j** (appears 2 times):
  - Line 3876, Column 39
  - Line 3878, Column 77

- **k** (appears 2 times):
  - Line 3930, Column 53
  - Line 3930, Column 73

- **keyword** (appears 6 times):
  - Line 4399, Column 23
  - Line 4401, Column 25
  - Line 4494, Column 27
  - Line 4501, Column 27
  - Line 4403, Column 25
  - ... and 1 more occurrences

- **lookback** (appears 4 times):
  - Line 5002, Column 44
  - Line 4984, Column 63
  - Line 5005, Column 49
  - Line 5006, Column 82

- **metadata** (appears 1 times):
  - Line 6293, Column 60

- **model_error** (appears 1 times):
  - Line 5229, Column 89

- **model_name** (appears 4 times):
  - Line 5293, Column 35
  - Line 5226, Column 52
  - Line 5221, Column 39
  - Line 5229, Column 63

- **p** (appears 5 times):
  - Line 3089, Column 37
  - Line 5953, Column 48
  - Line 3089, Column 59
  - Line 3089, Column 115
  - Line 5861, Column 54

- **period** (appears 29 times):
  - Line 5764, Column 27
  - Line 5739, Column 36
  - Line 5744, Column 23
  - Line 5932, Column 43
  - Line 5745, Column 40
  - ... and 24 more occurrences

- **quality_error** (appears 1 times):
  - Line 2839, Column 65

- **r** (appears 2 times):
  - Line 6584, Column 88
  - Line 6585, Column 88

- **recovery_error** (appears 1 times):
  - Line 2822, Column 71

- **s** (appears 2 times):
  - Line 5954, Column 65
  - Line 5861, Column 60

- **splits** (appears 3 times):
  - Line 4789, Column 15
  - Line 4785, Column 12
  - Line 4788, Column 40

- **symbol** (appears 6 times):
  - Line 6293, Column 22
  - Line 6320, Column 16
  - Line 6360, Column 22
  - Line 6291, Column 56
  - Line 6318, Column 56
  - ... and 1 more occurrences

- **timestamp** (appears 2 times):
  - Line 1966, Column 31
  - Line 1968, Column 80

- **type_interactions** (appears 3 times):
  - Line 3973, Column 33
  - Line 3970, Column 45
  - Line 3976, Column 57

- **x** (appears 7 times):
  - Line 5761, Column 69
  - Line 3946, Column 47
  - Line 5917, Column 30
  - Line 2361, Column 50
  - Line 3981, Column 54
  - ... and 2 more occurrences

---

### 3. src/training/steps/market_analysis/optimal_regime_clustering_backup/orchestrator.py

**Total Issues:** 437

**Issues by Name:**

- **c** (appears 50 times):
  - Line 2817, Column 23
  - Line 2818, Column 24
  - Line 2819, Column 33
  - Line 2905, Column 32
  - Line 3103, Column 35
  - ... and 45 more occurrences

- **cluster** (appears 37 times):
  - Line 2557, Column 25
  - Line 2558, Column 33
  - Line 2573, Column 25
  - Line 2574, Column 33
  - Line 2765, Column 16
  - ... and 32 more occurrences

- **cluster_i** (appears 4 times):
  - Line 2859, Column 38
  - Line 2860, Column 36
  - Line 2871, Column 41
  - Line 2863, Column 88

- **cluster_id** (appears 8 times):
  - Line 1207, Column 31
  - Line 1566, Column 36
  - Line 3211, Column 35
  - Line 3511, Column 34
  - Line 2498, Column 38
  - ... and 3 more occurrences

- **cluster_j** (appears 4 times):
  - Line 2859, Column 60
  - Line 2860, Column 56
  - Line 2872, Column 41
  - Line 2863, Column 107

- **cluster_metrics** (appears 24 times):
  - Line 1567, Column 40
  - Line 1568, Column 75
  - Line 1569, Column 77
  - Line 1570, Column 79
  - Line 1710, Column 19
  - ... and 19 more occurrences

- **col** (appears 6 times):
  - Line 439, Column 32
  - Line 511, Column 35
  - Line 505, Column 43
  - Line 511, Column 69
  - Line 439, Column 82
  - ... and 1 more occurrences

- **d** (appears 2 times):
  - Line 1900, Column 75
  - Line 1900, Column 133

- **e** (appears 73 times):
  - Line 85, Column 53
  - Line 290, Column 29
  - Line 777, Column 33
  - Line 813, Column 33
  - Line 851, Column 33
  - ... and 68 more occurrences

- **f** (appears 7 times):
  - Line 883, Column 46
  - Line 889, Column 56
  - Line 895, Column 62
  - Line 1067, Column 39
  - Line 1181, Column 48
  - ... and 2 more occurrences

- **feature** (appears 1 times):
  - Line 439, Column 63

- **file** (appears 4 times):
  - Line 471, Column 23
  - Line 472, Column 58
  - Line 345, Column 31
  - Line 346, Column 66

- **file_path** (appears 5 times):
  - Line 482, Column 80
  - Line 355, Column 99
  - Line 486, Column 65
  - Line 358, Column 74
  - Line 360, Column 72

- **i** (appears 45 times):
  - Line 1021, Column 19
  - Line 2496, Column 19
  - Line 3085, Column 41
  - Line 3088, Column 42
  - Line 3571, Column 25
  - ... and 40 more occurrences

- **idx** (appears 1 times):
  - Line 2683, Column 50

- **impact** (appears 1 times):
  - Line 2323, Column 50

- **j** (appears 9 times):
  - Line 2857, Column 28
  - Line 2869, Column 27
  - Line 3012, Column 56
  - Line 2863, Column 46
  - Line 2869, Column 59
  - ... and 4 more occurrences

- **k** (appears 8 times):
  - Line 591, Column 24
  - Line 581, Column 69
  - Line 582, Column 31
  - Line 589, Column 67
  - Line 584, Column 37
  - ... and 3 more occurrences

- **keyword** (appears 1 times):
  - Line 504, Column 23

- **kwargs** (appears 7 times):
  - Line 3900, Column 102
  - Line 3944, Column 102
  - Line 3988, Column 102
  - Line 4035, Column 94
  - Line 293, Column 30
  - ... and 2 more occurrences

- **label** (appears 13 times):
  - Line 832, Column 51
  - Line 933, Column 51
  - Line 938, Column 49
  - Line 1020, Column 51
  - Line 1293, Column 51
  - ... and 8 more occurrences

- **m** (appears 16 times):
  - Line 1785, Column 35
  - Line 2152, Column 29
  - Line 2160, Column 27
  - Line 2457, Column 29
  - Line 1440, Column 29
  - ... and 11 more occurrences

- **mc** (appears 1 times):
  - Line 2679, Column 23

- **metric_name** (appears 4 times):
  - Line 3263, Column 22
  - Line 3311, Column 25
  - Line 3430, Column 33
  - Line 3471, Column 34

- **p** (appears 21 times):
  - Line 2717, Column 38
  - Line 2718, Column 38
  - Line 2723, Column 39
  - Line 2724, Column 42
  - Line 2734, Column 39
  - ... and 16 more occurrences

- **path** (appears 6 times):
  - Line 416, Column 31
  - Line 418, Column 98
  - Line 420, Column 33
  - Line 417, Column 40
  - Line 421, Column 40
  - ... and 1 more occurrences

- **pattern** (appears 1 times):
  - Line 57, Column 30

- **r** (appears 25 times):
  - Line 1811, Column 32
  - Line 1812, Column 32
  - Line 2355, Column 47
  - Line 2637, Column 33
  - Line 3346, Column 29
  - ... and 20 more occurrences

- **root** (appears 2 times):
  - Line 472, Column 50
  - Line 346, Column 58

- **rs** (appears 4 times):
  - Line 2817, Column 85
  - Line 2818, Column 86
  - Line 2819, Column 103
  - Line 2823, Column 49

- **s** (appears 17 times):
  - Line 3345, Column 27
  - Line 3823, Column 33
  - Line 3827, Column 31
  - Line 3828, Column 33
  - Line 3829, Column 30
  - ... and 12 more occurrences

- **scenario** (appears 1 times):
  - Line 2113, Column 31

- **shock** (appears 2 times):
  - Line 2112, Column 130
  - Line 2112, Column 61

- **v** (appears 23 times):
  - Line 2055, Column 32
  - Line 2056, Column 35
  - Line 2057, Column 33
  - Line 3591, Column 24
  - Line 3355, Column 29
  - ... and 18 more occurrences

- **volume** (appears 2 times):
  - Line 2317, Column 19
  - Line 2319, Column 40

- **w** (appears 2 times):
  - Line 2425, Column 38
  - Line 2429, Column 38

---

### 4. research/candle_based_features/advanced_candle_features.py

**Total Issues:** 372

**Issues by Name:**

- **col** (appears 1 times):
  - Line 182, Column 19

- **i** (appears 363 times):
  - Line 315, Column 16
  - Line 324, Column 31
  - Line 583, Column 28
  - Line 596, Column 15
  - Line 656, Column 27
  - ... and 358 more occurrences

- **level** (appears 2 times):
  - Line 1746, Column 66
  - Line 1751, Column 63

- **pattern** (appears 2 times):
  - Line 1109, Column 35
  - Line 1109, Column 65

- **tf_period** (appears 4 times):
  - Line 628, Column 36
  - Line 630, Column 82
  - Line 634, Column 86
  - Line 638, Column 86

---

### 5. research/candle_ml_patterns/advanced_candle_features.py

**Total Issues:** 372

**Issues by Name:**

- **col** (appears 1 times):
  - Line 182, Column 19

- **i** (appears 363 times):
  - Line 315, Column 16
  - Line 324, Column 31
  - Line 583, Column 28
  - Line 596, Column 15
  - Line 656, Column 27
  - ... and 358 more occurrences

- **level** (appears 2 times):
  - Line 1746, Column 66
  - Line 1751, Column 63

- **pattern** (appears 2 times):
  - Line 1109, Column 35
  - Line 1109, Column 65

- **tf_period** (appears 4 times):
  - Line 628, Column 36
  - Line 630, Column 82
  - Line 634, Column 86
  - Line 638, Column 86

---

### 6. src/training/steps/backtesting/final_parameters_optimization.py

**Total Issues:** 337

**Issues by Name:**

- **best_coarse_params** (appears 1 times):
  - Line 4499, Column 71

- **best_grid_params** (appears 1 times):
  - Line 4548, Column 78

- **candidate** (appears 4 times):
  - Line 4014, Column 19
  - Line 4017, Column 55
  - Line 4015, Column 25
  - Line 4020, Column 86

- **combination** (appears 2 times):
  - Line 4653, Column 30
  - Line 4707, Column 30

- **data_dir** (appears 3 times):
  - Line 4792, Column 83
  - Line 4805, Column 86
  - Line 4309, Column 34

- **e** (appears 72 times):
  - Line 88, Column 59
  - Line 4378, Column 33
  - Line 4431, Column 75
  - Line 311, Column 77
  - Line 312, Column 75
  - ... and 67 more occurrences

- **exc** (appears 2 times):
  - Line 4241, Column 90
  - Line 4020, Column 99

- **exchange** (appears 7 times):
  - Line 4792, Column 73
  - Line 4805, Column 76
  - Line 4253, Column 49
  - Line 4258, Column 46
  - Line 4310, Column 50
  - ... and 2 more occurrences

- **f** (appears 4 times):
  - Line 4256, Column 50
  - Line 4261, Column 48
  - Line 4296, Column 52
  - Line 4319, Column 42

- **fold_idx** (appears 3 times):
  - Line 670, Column 32
  - Line 664, Column 64
  - Line 678, Column 52

- **fp** (appears 1 times):
  - Line 4016, Column 42

- **i** (appears 17 times):
  - Line 3109, Column 39
  - Line 1053, Column 44
  - Line 1620, Column 75
  - Line 3109, Column 41
  - Line 1053, Column 82
  - ... and 12 more occurrences

- **key** (appears 19 times):
  - Line 3889, Column 15
  - Line 3893, Column 15
  - Line 1051, Column 30
  - Line 1298, Column 25
  - Line 1304, Column 25
  - ... and 14 more occurrences

- **long_key** (appears 2 times):
  - Line 1309, Column 15
  - Line 1310, Column 27

- **long_value** (appears 1 times):
  - Line 1313, Column 48

- **param** (appears 15 times):
  - Line 3494, Column 30
  - Line 2732, Column 19
  - Line 2733, Column 37
  - Line 2743, Column 19
  - Line 2793, Column 19
  - ... and 10 more occurrences

- **param_config** (appears 79 times):
  - Line 4437, Column 35
  - Line 4719, Column 45
  - Line 1721, Column 15
  - Line 4438, Column 33
  - Line 4621, Column 15
  - ... and 74 more occurrences

- **param_name** (appears 45 times):
  - Line 4665, Column 15
  - Line 4668, Column 37
  - Line 4718, Column 15
  - Line 4722, Column 37
  - Line 4739, Column 27
  - ... and 40 more occurrences

- **path_candidates** (appears 4 times):
  - Line 4012, Column 25
  - Line 4007, Column 8
  - Line 3997, Column 12
  - Line 4005, Column 12

- **profit_factors** (appears 3 times):
  - Line 4046, Column 62
  - Line 4038, Column 12
  - Line 4046, Column 42

- **return_val** (appears 2 times):
  - Line 2651, Column 23
  - Line 2654, Column 25

- **rr_values** (appears 3 times):
  - Line 4047, Column 46
  - Line 4039, Column 12
  - Line 4047, Column 31

- **should_exit** (appears 1 times):
  - Line 2649, Column 19

- **space** (appears 1 times):
  - Line 1353, Column 69

- **symbol** (appears 7 times):
  - Line 4792, Column 65
  - Line 4805, Column 68
  - Line 4253, Column 60
  - Line 4258, Column 57
  - Line 4310, Column 61
  - ... and 2 more occurrences

- **t** (appears 2 times):
  - Line 4386, Column 22
  - Line 4386, Column 55

- **train_idx** (appears 3 times):
  - Line 646, Column 47
  - Line 647, Column 46
  - Line 673, Column 42

- **v** (appears 8 times):
  - Line 4631, Column 56
  - Line 4686, Column 56
  - Line 4640, Column 56
  - Line 4694, Column 56
  - Line 4643, Column 56
  - ... and 3 more occurrences

- **val_idx** (appears 3 times):
  - Line 650, Column 47
  - Line 651, Column 46
  - Line 674, Column 40

- **value** (appears 15 times):
  - Line 1296, Column 42
  - Line 1298, Column 32
  - Line 1302, Column 43
  - Line 1304, Column 32
  - Line 1050, Column 26
  - ... and 10 more occurrences

- **win_rates** (appears 6 times):
  - Line 4041, Column 15
  - Line 4044, Column 32
  - Line 4045, Column 31
  - Line 4037, Column 12
  - Line 4048, Column 60
  - ... and 1 more occurrences

- **x** (appears 1 times):
  - Line 4558, Column 47

---

### 7. exchanges/binance.py

**Total Issues:** 337

**Issues by Name:**

- **balance** (appears 2 times):
  - Line 1250, Column 27
  - Line 1249, Column 19

- **callback** (appears 4 times):
  - Line 1161, Column 26
  - Line 1175, Column 26
  - Line 1189, Column 26
  - Line 1203, Column 26

- **client_order_id** (appears 3 times):
  - Line 670, Column 11
  - Line 654, Column 55
  - Line 671, Column 47

- **currency** (appears 3 times):
  - Line 1251, Column 29
  - Line 1249, Column 55
  - Line 1253, Column 60

- **e** (appears 17 times):
  - Line 157, Column 61
  - Line 210, Column 61
  - Line 240, Column 53
  - Line 267, Column 65
  - Line 292, Column 45
  - ... and 12 more occurrences

- **end_time** (appears 2 times):
  - Line 572, Column 25
  - Line 580, Column 31

- **end_time_ms** (appears 4 times):
  - Line 905, Column 28
  - Line 912, Column 23
  - Line 950, Column 28
  - Line 956, Column 23

- **futures** (appears 2 times):
  - Line 801, Column 46
  - Line 820, Column 82

- **instrument** (appears 17 times):
  - Line 340, Column 19
  - Line 319, Column 27
  - Line 320, Column 36
  - Line 321, Column 39
  - Line 322, Column 30
  - ... and 12 more occurrences

- **interval** (appears 11 times):
  - Line 536, Column 15
  - Line 544, Column 24
  - Line 568, Column 15
  - Line 578, Column 24
  - Line 851, Column 15
  - ... and 6 more occurrences

- **item** (appears 46 times):
  - Line 867, Column 30
  - Line 867, Column 50
  - Line 922, Column 33
  - Line 923, Column 33
  - Line 924, Column 28
  - ... and 41 more occurrences

- **limit** (appears 28 times):
  - Line 498, Column 11
  - Line 498, Column 25
  - Line 502, Column 57
  - Line 517, Column 11
  - Line 517, Column 25
  - ... and 23 more occurrences

- **method** (appears 2 times):
  - Line 813, Column 40
  - Line 820, Column 48

- **order_id** (appears 10 times):
  - Line 698, Column 15
  - Line 706, Column 29
  - Line 719, Column 15
  - Line 727, Column 29
  - Line 1050, Column 15
  - ... and 5 more occurrences

- **order_type** (appears 6 times):
  - Line 643, Column 15
  - Line 654, Column 26
  - Line 995, Column 15
  - Line 1221, Column 58
  - Line 661, Column 20
  - ... and 1 more occurrences

- **position** (appears 2 times):
  - Line 1029, Column 27
  - Line 1028, Column 19

- **price** (appears 10 times):
  - Line 654, Column 48
  - Line 666, Column 11
  - Line 1009, Column 11
  - Line 1010, Column 36
  - Line 647, Column 11
  - ... and 5 more occurrences

- **quantity** (appears 6 times):
  - Line 645, Column 11
  - Line 654, Column 38
  - Line 997, Column 11
  - Line 1006, Column 24
  - Line 662, Column 28
  - ... and 1 more occurrences

- **response** (appears 74 times):
  - Line 415, Column 15
  - Line 432, Column 15
  - Line 454, Column 15
  - Line 468, Column 15
  - Line 485, Column 15
  - ... and 69 more occurrences

- **side** (appears 8 times):
  - Line 654, Column 20
  - Line 641, Column 15
  - Line 993, Column 15
  - Line 1221, Column 52
  - Line 660, Column 20
  - ... and 3 more occurrences

- **signed** (appears 2 times):
  - Line 808, Column 11
  - Line 820, Column 74

- **start_time** (appears 2 times):
  - Line 572, Column 11
  - Line 579, Column 33

- **start_time_ms** (appears 4 times):
  - Line 905, Column 11
  - Line 911, Column 25
  - Line 950, Column 11
  - Line 955, Column 25

- **stop_price** (appears 4 times):
  - Line 668, Column 11
  - Line 649, Column 11
  - Line 649, Column 38
  - Line 669, Column 44

- **symbol** (appears 65 times):
  - Line 748, Column 11
  - Line 479, Column 15
  - Line 496, Column 15
  - Line 515, Column 15
  - Line 534, Column 15
  - ... and 60 more occurrences

- **symbol_info** (appears 2 times):
  - Line 1150, Column 27
  - Line 1149, Column 19

- **trade** (appears 1 times):
  - Line 1189, Column 35

---

### 8. src/trading/reporting/performance_reporter.py

**Total Issues:** 304

**Issues by Name:**

- **action** (appears 2 times):
  - Line 205, Column 42
  - Line 201, Column 66

- **data** (appears 11 times):
  - Line 437, Column 35
  - Line 451, Column 34
  - Line 441, Column 20
  - Line 442, Column 20
  - Line 443, Column 20
  - ... and 6 more occurrences

- **e** (appears 15 times):
  - Line 101, Column 33
  - Line 100, Column 73
  - Line 182, Column 70
  - Line 232, Column 67
  - Line 284, Column 79
  - ... and 10 more occurrences

- **f** (appears 2 times):
  - Line 650, Column 34
  - Line 756, Column 16

- **feature** (appears 8 times):
  - Line 310, Column 20
  - Line 319, Column 23
  - Line 323, Column 47
  - Line 301, Column 41
  - Line 302, Column 27
  - ... and 3 more occurrences

- **features** (appears 1 times):
  - Line 311, Column 48

- **importance** (appears 1 times):
  - Line 304, Column 82

- **key** (appears 6 times):
  - Line 688, Column 36
  - Line 679, Column 47
  - Line 683, Column 42
  - Line 772, Column 51
  - Line 772, Column 105
  - ... and 1 more occurrences

- **max_conf** (appears 2 times):
  - Line 615, Column 57
  - Line 630, Column 59

- **max_size** (appears 2 times):
  - Line 582, Column 53
  - Line 589, Column 49

- **metric_name** (appears 1 times):
  - Line 695, Column 52

- **metrics** (appears 12 times):
  - Line 694, Column 46
  - Line 792, Column 25
  - Line 793, Column 97
  - Line 794, Column 25
  - Line 795, Column 25
  - ... and 7 more occurrences

- **min_conf** (appears 2 times):
  - Line 615, Column 23
  - Line 630, Column 48

- **min_size** (appears 2 times):
  - Line 582, Column 23
  - Line 589, Column 39

- **model_id** (appears 17 times):
  - Line 309, Column 39
  - Line 143, Column 32
  - Line 263, Column 38
  - Line 297, Column 23
  - Line 791, Column 25
  - ... and 12 more occurrences

- **p** (appears 5 times):
  - Line 209, Column 41
  - Line 400, Column 54
  - Line 592, Column 41
  - Line 209, Column 66
  - Line 592, Column 68

- **quartile** (appears 1 times):
  - Line 587, Column 49

- **report_name** (appears 11 times):
  - Line 106, Column 27
  - Line 837, Column 93
  - Line 93, Column 46
  - Line 714, Column 53
  - Line 731, Column 49
  - ... and 6 more occurrences

- **session_metrics** (appears 2 times):
  - Line 837, Column 76
  - Line 82, Column 84

- **shap_values** (appears 1 times):
  - Line 300, Column 47

- **t** (appears 87 times):
  - Line 122, Column 34
  - Line 123, Column 33
  - Line 127, Column 26
  - Line 146, Column 33
  - Line 201, Column 33
  - ... and 82 more occurrences

- **trade** (appears 25 times):
  - Line 415, Column 25
  - Line 426, Column 60
  - Line 191, Column 36
  - Line 196, Column 36
  - Line 364, Column 19
  - ... and 20 more occurrences

- **trades** (appears 74 times):
  - Line 141, Column 25
  - Line 190, Column 25
  - Line 195, Column 25
  - Line 242, Column 25
  - Line 294, Column 25
  - ... and 69 more occurrences

- **value** (appears 9 times):
  - Line 688, Column 45
  - Line 679, Column 56
  - Line 683, Column 51
  - Line 695, Column 69
  - Line 773, Column 63
  - ... and 4 more occurrences

- **x** (appears 5 times):
  - Line 326, Column 84
  - Line 452, Column 64
  - Line 453, Column 65
  - Line 169, Column 78
  - Line 545, Column 88

---

### 9. src/monitoring/csv_export_manager.py

**Total Issues:** 286

**Issues by Name:**

- **Any** (appears 12 times):
  - Line 871, Column 44
  - Line 67, Column 65
  - Line 90, Column 74
  - Line 158, Column 75
  - Line 203, Column 76
  - ... and 7 more occurrences

- **Dict** (appears 2 times):
  - Line 871, Column 34
  - Line 50, Column 31

- **List** (appears 11 times):
  - Line 61, Column 29
  - Line 67, Column 60
  - Line 90, Column 69
  - Line 158, Column 70
  - Line 203, Column 71
  - ... and 6 more occurrences

- **Optional** (appears 6 times):
  - Line 43, Column 23
  - Line 850, Column 57
  - Line 62, Column 31
  - Line 68, Column 43
  - Line 545, Column 43
  - ... and 1 more occurrences

- **action** (appears 2 times):
  - Line 467, Column 40
  - Line 469, Column 48

- **col** (appears 5 times):
  - Line 697, Column 30
  - Line 697, Column 59
  - Line 697, Column 89
  - Line 699, Column 30
  - Line 702, Column 32

- **count** (appears 7 times):
  - Line 594, Column 55
  - Line 459, Column 29
  - Line 468, Column 29
  - Line 477, Column 29
  - Line 807, Column 29
  - ... and 2 more occurrences

- **daily_summaries** (appears 2 times):
  - Line 557, Column 27
  - Line 551, Column 19

- **dataclass** (appears 2 times):
  - Line 22, Column 1
  - Line 34, Column 1

- **day** (appears 2 times):
  - Line 529, Column 41
  - Line 531, Column 65

- **decision** (appears 59 times):
  - Line 254, Column 23
  - Line 289, Column 15
  - Line 100, Column 41
  - Line 210, Column 31
  - Line 219, Column 26
  - ... and 54 more occurrences

- **decisions** (appears 5 times):
  - Line 110, Column 66
  - Line 122, Column 64
  - Line 123, Column 54
  - Line 124, Column 51
  - Line 125, Column 52

- **description** (appears 1 times):
  - Line 864, Column 42

- **e** (appears 13 times):
  - Line 87, Column 66
  - Line 155, Column 74
  - Line 200, Column 75
  - Line 328, Column 79
  - Line 364, Column 69
  - ... and 8 more occurrences

- **f** (appears 2 times):
  - Line 135, Column 73
  - Line 135, Column 29

- **feature** (appears 1 times):
  - Line 394, Column 65

- **filename** (appears 1 times):
  - Line 853, Column 42

- **hour** (appears 2 times):
  - Line 520, Column 42
  - Line 522, Column 70

- **i** (appears 13 times):
  - Line 271, Column 33
  - Line 272, Column 33
  - Line 273, Column 33
  - Line 274, Column 33
  - Line 275, Column 33
  - ... and 8 more occurrences

- **importance** (appears 1 times):
  - Line 394, Column 84

- **indicator** (appears 12 times):
  - Line 271, Column 45
  - Line 276, Column 52
  - Line 272, Column 52
  - Line 273, Column 53
  - Line 274, Column 57
  - ... and 7 more occurrences

- **key** (appears 6 times):
  - Line 249, Column 38
  - Line 251, Column 38
  - Line 292, Column 39
  - Line 294, Column 39
  - Line 429, Column 55
  - ... and 1 more occurrences

- **metric** (appears 13 times):
  - Line 815, Column 19
  - Line 818, Column 41
  - Line 820, Column 51
  - Line 823, Column 41
  - Line 825, Column 68
  - ... and 8 more occurrences

- **model_decision** (appears 17 times):
  - Line 280, Column 39
  - Line 286, Column 44
  - Line 281, Column 41
  - Line 282, Column 53
  - Line 283, Column 53
  - ... and 12 more occurrences

- **model_id** (appears 2 times):
  - Line 267, Column 36
  - Line 355, Column 48

- **model_performances** (appears 2 times):
  - Line 729, Column 24
  - Line 723, Column 19

- **model_type** (appears 2 times):
  - Line 806, Column 44
  - Line 808, Column 48

- **perf** (appears 17 times):
  - Line 731, Column 32
  - Line 732, Column 34
  - Line 734, Column 38
  - Line 735, Column 39
  - Line 736, Column 36
  - ... and 12 more occurrences

- **prob** (appears 1 times):
  - Line 243, Column 80

- **regime_id** (appears 2 times):
  - Line 594, Column 34
  - Line 243, Column 47

- **separate_by_mode** (appears 1 times):
  - Line 79, Column 15

- **summary** (appears 35 times):
  - Line 560, Column 36
  - Line 561, Column 36
  - Line 562, Column 35
  - Line 563, Column 36
  - Line 564, Column 35
  - ... and 30 more occurrences

- **token** (appears 2 times):
  - Line 476, Column 39
  - Line 478, Column 62

- **trade_decisions** (appears 14 times):
  - Line 207, Column 24
  - Line 96, Column 28
  - Line 305, Column 28
  - Line 335, Column 28
  - Line 371, Column 28
  - ... and 9 more occurrences

- **value** (appears 9 times):
  - Line 248, Column 34
  - Line 291, Column 34
  - Line 249, Column 53
  - Line 251, Column 51
  - Line 292, Column 54
  - ... and 4 more occurrences

- **weight** (appears 2 times):
  - Line 267, Column 56
  - Line 355, Column 68

---

### 10. exchanges/okx.py

**Total Issues:** 270

**Issues by Name:**

- **candle** (appears 7 times):
  - Line 986, Column 23
  - Line 987, Column 26
  - Line 988, Column 21
  - Line 989, Column 21
  - Line 990, Column 20
  - ... and 2 more occurrences

- **client_order_id** (appears 3 times):
  - Line 702, Column 59
  - Line 1037, Column 28
  - Line 712, Column 27

- **currency** (appears 1 times):
  - Line 1060, Column 51

- **e** (appears 36 times):
  - Line 781, Column 51
  - Line 209, Column 72
  - Line 347, Column 60
  - Line 356, Column 57
  - Line 380, Column 61
  - ... and 31 more occurrences

- **end_time** (appears 7 times):
  - Line 1185, Column 26
  - Line 1188, Column 30
  - Line 1216, Column 26
  - Line 1219, Column 30
  - Line 1186, Column 34
  - ... and 2 more occurrences

- **instrument** (appears 17 times):
  - Line 267, Column 15
  - Line 250, Column 23
  - Line 251, Column 32
  - Line 252, Column 35
  - Line 253, Column 26
  - ... and 12 more occurrences

- **k** (appears 12 times):
  - Line 530, Column 46
  - Line 530, Column 67
  - Line 530, Column 88
  - Line 531, Column 40
  - Line 531, Column 62
  - ... and 7 more occurrences

- **limit** (appears 10 times):
  - Line 980, Column 80
  - Line 441, Column 58
  - Line 457, Column 61
  - Line 1160, Column 68
  - Line 1191, Column 62
  - ... and 5 more occurrences

- **method** (appears 1 times):
  - Line 929, Column 15

- **order_id** (appears 18 times):
  - Line 787, Column 73
  - Line 792, Column 25
  - Line 855, Column 57
  - Line 752, Column 77
  - Line 757, Column 29
  - ... and 13 more occurrences

- **order_type** (appears 9 times):
  - Line 1012, Column 26
  - Line 644, Column 30
  - Line 702, Column 30
  - Line 1238, Column 64
  - Line 1028, Column 46
  - ... and 4 more occurrences

- **position** (appears 14 times):
  - Line 1129, Column 15
  - Line 1094, Column 19
  - Line 1112, Column 41
  - Line 1113, Column 39
  - Line 1114, Column 41
  - ... and 9 more occurrences

- **response** (appears 49 times):
  - Line 337, Column 19
  - Line 371, Column 19
  - Line 395, Column 19
  - Line 411, Column 19
  - Line 427, Column 19
  - ... and 44 more occurrences

- **side** (appears 7 times):
  - Line 1012, Column 20
  - Line 644, Column 24
  - Line 702, Column 24
  - Line 1238, Column 58
  - Line 1027, Column 38
  - ... and 2 more occurrences

- **start_time** (appears 7 times):
  - Line 1180, Column 26
  - Line 1183, Column 32
  - Line 1211, Column 26
  - Line 1214, Column 32
  - Line 1181, Column 36
  - ... and 2 more occurrences

- **stop_price** (appears 3 times):
  - Line 718, Column 15
  - Line 1036, Column 23
  - Line 719, Column 50

- **symbol** (appears 67 times):
  - Line 882, Column 15
  - Line 1012, Column 12
  - Line 1022, Column 66
  - Line 502, Column 15
  - Line 644, Column 16
  - ... and 62 more occurrences

- **timeframe** (appears 2 times):
  - Line 977, Column 35
  - Line 993, Column 25

---

### 11. src/training/steps/market_analysis/clusters/iterative_optimization.py

**Total Issues:** 269

**Issues by Name:**

- **alt** (appears 5 times):
  - Line 2447, Column 26
  - Line 2447, Column 38
  - Line 2448, Column 39
  - Line 2449, Column 27
  - Line 4929, Column 39

- **b_idx** (appears 1 times):
  - Line 6814, Column 36

- **candidate_cluster** (appears 1 times):
  - Line 1048, Column 56

- **chunk** (appears 6 times):
  - Line 316, Column 42
  - Line 317, Column 48
  - Line 358, Column 42
  - Line 359, Column 48
  - Line 313, Column 23
  - ... and 1 more occurrences

- **chunk_i** (appears 3 times):
  - Line 451, Column 87
  - Line 452, Column 47
  - Line 454, Column 60

- **chunk_j** (appears 3 times):
  - Line 451, Column 106
  - Line 452, Column 56
  - Line 454, Column 51

- **cidj** (appears 1 times):
  - Line 2457, Column 20

- **component** (appears 8 times):
  - Line 5135, Column 27
  - Line 5146, Column 31
  - Line 5148, Column 31
  - Line 5129, Column 38
  - Line 5130, Column 53
  - ... and 3 more occurrences

- **e** (appears 93 times):
  - Line 1085, Column 67
  - Line 215, Column 70
  - Line 262, Column 64
  - Line 273, Column 67
  - Line 300, Column 66
  - ... and 88 more occurrences

- **enable_risk_mitigation** (appears 1 times):
  - Line 4515, Column 15

- **end** (appears 2 times):
  - Line 7296, Column 28
  - Line 7295, Column 38

- **entity_id** (appears 1 times):
  - Line 3433, Column 37

- **err** (appears 2 times):
  - Line 2650, Column 53
  - Line 6430, Column 56

- **hw_error** (appears 1 times):
  - Line 205, Column 70

- **k_global** (appears 2 times):
  - Line 7920, Column 76
  - Line 7968, Column 113

- **key** (appears 3 times):
  - Line 8141, Column 35
  - Line 8140, Column 50
  - Line 8141, Column 52

- **kwargs** (appears 1 times):
  - Line 6424, Column 52

- **lab** (appears 9 times):
  - Line 6689, Column 24
  - Line 6696, Column 15
  - Line 6695, Column 42
  - Line 7310, Column 29
  - Line 6689, Column 52
  - ... and 4 more occurrences

- **large_cluster** (appears 3 times):
  - Line 7140, Column 43
  - Line 7136, Column 29
  - Line 7155, Column 30

- **moves** (appears 2 times):
  - Line 8456, Column 23
  - Line 8459, Column 49

- **neighbor_idx** (appears 1 times):
  - Line 8010, Column 46

- **new** (appears 1 times):
  - Line 1396, Column 45

- **new_label** (appears 1 times):
  - Line 6266, Column 32

- **old** (appears 2 times):
  - Line 5566, Column 17
  - Line 1396, Column 40

- **old_id** (appears 12 times):
  - Line 1209, Column 31
  - Line 1210, Column 47
  - Line 1325, Column 26
  - Line 1926, Column 31
  - Line 1927, Column 47
  - ... and 7 more occurrences

- **old_label** (appears 1 times):
  - Line 6266, Column 21

- **other_c** (appears 1 times):
  - Line 248, Column 48

- **other_cluster** (appears 11 times):
  - Line 3756, Column 67
  - Line 3764, Column 35
  - Line 3746, Column 19
  - Line 3750, Column 58
  - Line 8437, Column 19
  - ... and 6 more occurrences

- **other_cluster_id** (appears 4 times):
  - Line 944, Column 39
  - Line 938, Column 19
  - Line 938, Column 94
  - Line 941, Column 97

- **other_id** (appears 5 times):
  - Line 5844, Column 50
  - Line 5861, Column 30
  - Line 5841, Column 19
  - Line 5841, Column 38
  - Line 5855, Column 51

- **percentage** (appears 4 times):
  - Line 9135, Column 19
  - Line 9137, Column 21
  - Line 9139, Column 29
  - Line 9144, Column 47

- **r** (appears 3 times):
  - Line 2444, Column 25
  - Line 2446, Column 36
  - Line 2448, Column 24

- **regime** (appears 2 times):
  - Line 3348, Column 72
  - Line 3367, Column 80

- **rel_i** (appears 1 times):
  - Line 6814, Column 23

- **report** (appears 1 times):
  - Line 4737, Column 37

- **round_num** (appears 21 times):
  - Line 4578, Column 36
  - Line 4581, Column 49
  - Line 4585, Column 44
  - Line 4591, Column 24
  - Line 4598, Column 50
  - ... and 16 more occurrences

- **seed** (appears 3 times):
  - Line 7078, Column 109
  - Line 3697, Column 43
  - Line 7096, Column 48

- **sk_error** (appears 1 times):
  - Line 195, Column 72

- **small_cluster** (appears 5 times):
  - Line 7154, Column 45
  - Line 7156, Column 30
  - Line 7145, Column 105
  - Line 7158, Column 33
  - Line 7145, Column 57

- **source_cluster** (appears 1 times):
  - Line 5343, Column 68

- **sub_cluster** (appears 2 times):
  - Line 8494, Column 46
  - Line 8497, Column 73

- **sync_error** (appears 1 times):
  - Line 2946, Column 41

- **t** (appears 14 times):
  - Line 2459, Column 23
  - Line 8092, Column 54
  - Line 8108, Column 39
  - Line 6690, Column 40
  - Line 8095, Column 51
  - ... and 9 more occurrences

- **target** (appears 3 times):
  - Line 7982, Column 15
  - Line 7983, Column 35
  - Line 7984, Column 25

- **ve** (appears 3 times):
  - Line 8840, Column 39
  - Line 8840, Column 25
  - Line 8673, Column 56

- **x** (appears 17 times):
  - Line 982, Column 33
  - Line 3001, Column 38
  - Line 907, Column 48
  - Line 3609, Column 41
  - Line 6581, Column 61
  - ... and 12 more occurrences

---

### 12. src/training/steps/market_analysis/tas_regime/core/tas_regime_detector.py

**Total Issues:** 267

**Issues by Name:**

- **best_params_collection** (appears 1 times):
  - Line 2337, Column 16

- **cache_error** (appears 1 times):
  - Line 747, Column 78

- **cache_store_error** (appears 1 times):
  - Line 985, Column 94

- **chunk_error** (appears 1 times):
  - Line 1151, Column 87

- **chunk_predictions** (appears 3 times):
  - Line 1177, Column 15
  - Line 1178, Column 55
  - Line 1168, Column 20

- **chunk_probabilities** (appears 2 times):
  - Line 1179, Column 57
  - Line 1169, Column 20

- **clustering_error** (appears 2 times):
  - Line 801, Column 60
  - Line 803, Column 59

- **count** (appears 8 times):
  - Line 816, Column 30
  - Line 1527, Column 30
  - Line 1650, Column 30
  - Line 2669, Column 22
  - Line 817, Column 47
  - ... and 3 more occurrences

- **detection_results** (appears 13 times):
  - Line 1233, Column 15
  - Line 1236, Column 8
  - Line 1241, Column 24
  - Line 1242, Column 16
  - Line 1255, Column 26
  - ... and 8 more occurrences

- **df** (appears 2 times):
  - Line 758, Column 43
  - Line 758, Column 89

- **e** (appears 79 times):
  - Line 2153, Column 33
  - Line 2257, Column 33
  - Line 2303, Column 33
  - Line 2388, Column 33
  - Line 2457, Column 33
  - ... and 74 more occurrences

- **executor** (appears 1 times):
  - Line 1226, Column 20

- **f** (appears 8 times):
  - Line 3179, Column 36
  - Line 3191, Column 37
  - Line 2370, Column 48
  - Line 2371, Column 49
  - Line 2372, Column 50
  - ... and 3 more occurrences

- **fold_idx** (appears 2 times):
  - Line 2356, Column 28
  - Line 2435, Column 28

- **future** (appears 1 times):
  - Line 1230, Column 29

- **group** (appears 2 times):
  - Line 1081, Column 30
  - Line 1085, Column 55

- **i** (appears 43 times):
  - Line 1916, Column 40
  - Line 1917, Column 35
  - Line 1932, Column 33
  - Line 2472, Column 40
  - Line 1707, Column 39
  - ... and 38 more occurrences

- **idx** (appears 1 times):
  - Line 1127, Column 36

- **item** (appears 2 times):
  - Line 1226, Column 55
  - Line 2375, Column 59

- **j** (appears 2 times):
  - Line 3159, Column 23
  - Line 3160, Column 48

- **k** (appears 5 times):
  - Line 661, Column 20
  - Line 1742, Column 39
  - Line 2167, Column 31
  - Line 2167, Column 78
  - Line 2730, Column 80

- **key** (appears 3 times):
  - Line 2181, Column 15
  - Line 2187, Column 23
  - Line 2183, Column 27

- **label** (appears 1 times):
  - Line 2868, Column 40

- **large_regime** (appears 3 times):
  - Line 2752, Column 43
  - Line 2767, Column 50
  - Line 2767, Column 101

- **matrix_error** (appears 1 times):
  - Line 1114, Column 92

- **monitor_error** (appears 1 times):
  - Line 1202, Column 72

- **new_id** (appears 1 times):
  - Line 2773, Column 38

- **old_id** (appears 1 times):
  - Line 2773, Column 30

- **optimization_error** (appears 1 times):
  - Line 1125, Column 63

- **pred** (appears 2 times):
  - Line 3155, Column 40
  - Line 3159, Column 28

- **r** (appears 15 times):
  - Line 2710, Column 28
  - Line 2718, Column 29
  - Line 2713, Column 33
  - Line 2745, Column 33
  - Line 2710, Column 48
  - ... and 10 more occurrences

- **regime** (appears 2 times):
  - Line 3084, Column 45
  - Line 3108, Column 47

- **regime_id** (appears 6 times):
  - Line 1972, Column 44
  - Line 1969, Column 29
  - Line 2036, Column 48
  - Line 1970, Column 85
  - Line 2022, Column 33
  - ... and 1 more occurrences

- **rolling_matrices** (appears 4 times):
  - Line 2142, Column 15
  - Line 2143, Column 69
  - Line 2119, Column 16
  - Line 2114, Column 20

- **rolling_windows** (appears 3 times):
  - Line 2132, Column 35
  - Line 2120, Column 16
  - Line 2115, Column 20

- **scoring_metrics** (appears 1 times):
  - Line 2241, Column 24

- **size** (appears 1 times):
  - Line 2718, Column 70

- **small_regime** (appears 7 times):
  - Line 2722, Column 43
  - Line 2727, Column 58
  - Line 2735, Column 41
  - Line 2737, Column 45
  - Line 2734, Column 92
  - ... and 2 more occurrences

- **stack** (appears 3 times):
  - Line 1209, Column 12
  - Line 1212, Column 12
  - Line 1200, Column 20

- **start** (appears 3 times):
  - Line 2111, Column 26
  - Line 2112, Column 54
  - Line 2115, Column 44

- **start_idx** (appears 2 times):
  - Line 1160, Column 34
  - Line 1161, Column 43

- **sub_label** (appears 1 times):
  - Line 2764, Column 27

- **test_idx** (appears 8 times):
  - Line 2329, Column 38
  - Line 2329, Column 56
  - Line 2427, Column 47
  - Line 2325, Column 46
  - Line 2420, Column 46
  - ... and 3 more occurrences

- **tf** (appears 1 times):
  - Line 1127, Column 31

- **train_idx** (appears 10 times):
  - Line 2328, Column 40
  - Line 2328, Column 59
  - Line 2424, Column 31
  - Line 2424, Column 50
  - Line 2426, Column 48
  - ... and 5 more occurrences

- **ts** (appears 1 times):
  - Line 1134, Column 69

- **v** (appears 4 times):
  - Line 661, Column 47
  - Line 663, Column 44
  - Line 1742, Column 47
  - Line 2167, Column 34

- **warning** (appears 1 times):
  - Line 1357, Column 70

---

### 13. src/training/steps/pre_training/sub_pipeline.py

**Total Issues:** 262

**Issues by Name:**

- **alert** (appears 1 times):
  - Line 3280, Column 28

- **args** (appears 1 times):
  - Line 76, Column 25

- **attr** (appears 1 times):
  - Line 1077, Column 46

- **candidate** (appears 2 times):
  - Line 772, Column 30
  - Line 772, Column 72

- **col** (appears 9 times):
  - Line 3442, Column 36
  - Line 437, Column 41
  - Line 3442, Column 68
  - Line 3458, Column 40
  - Line 3457, Column 39
  - ... and 4 more occurrences

- **column** (appears 6 times):
  - Line 3155, Column 23
  - Line 3155, Column 41
  - Line 3350, Column 31
  - Line 3357, Column 42
  - Line 3341, Column 24
  - ... and 1 more occurrences

- **column_metrics** (appears 3 times):
  - Line 3365, Column 11
  - Line 3350, Column 12
  - Line 3367, Column 27

- **contract_error** (appears 6 times):
  - Line 3511, Column 89
  - Line 3687, Column 90
  - Line 4775, Column 90
  - Line 3508, Column 41
  - Line 3684, Column 41
  - ... and 1 more occurrences

- **e** (appears 94 times):
  - Line 3469, Column 101
  - Line 3645, Column 102
  - Line 3826, Column 107
  - Line 3972, Column 39
  - Line 4023, Column 39
  - ... and 89 more occurrences

- **err** (appears 3 times):
  - Line 1074, Column 30
  - Line 1074, Column 55
  - Line 1075, Column 27

- **event_extra** (appears 3 times):
  - Line 1495, Column 12
  - Line 1499, Column 12
  - Line 1509, Column 22

- **explicit** (appears 1 times):
  - Line 765, Column 12

- **extra** (appears 2 times):
  - Line 1336, Column 11
  - Line 1337, Column 27

- **ffs_duration_ms** (appears 1 times):
  - Line 2379, Column 32

- **flag** (appears 2 times):
  - Line 4850, Column 40
  - Line 4851, Column 23

- **formatted_errors** (appears 3 times):
  - Line 1119, Column 38
  - Line 1116, Column 16
  - Line 1118, Column 16

- **idx_error** (appears 1 times):
  - Line 3438, Column 95

- **include_disabled** (appears 2 times):
  - Line 1050, Column 15
  - Line 1044, Column 15

- **index** (appears 2 times):
  - Line 2212, Column 31
  - Line 2872, Column 48

- **interactive_context** (appears 5 times):
  - Line 2170, Column 12
  - Line 2171, Column 12
  - Line 2236, Column 24
  - Line 2255, Column 32
  - Line 2256, Column 33

- **issue** (appears 2 times):
  - Line 1410, Column 34
  - Line 1410, Column 73

- **item** (appears 9 times):
  - Line 1110, Column 26
  - Line 1112, Column 19
  - Line 1111, Column 58
  - Line 1111, Column 73
  - Line 1118, Column 44
  - ... and 4 more occurrences

- **k** (appears 8 times):
  - Line 3045, Column 31
  - Line 1429, Column 35
  - Line 1575, Column 37
  - Line 3045, Column 88
  - Line 3353, Column 37
  - ... and 3 more occurrences

- **key** (appears 26 times):
  - Line 609, Column 12
  - Line 609, Column 27
  - Line 1562, Column 16
  - Line 2785, Column 12
  - Line 2876, Column 18
  - ... and 21 more occurrences

- **kwargs** (appears 1 times):
  - Line 76, Column 33

- **merge_error** (appears 9 times):
  - Line 1259, Column 24
  - Line 1880, Column 38
  - Line 1262, Column 41
  - Line 1267, Column 30
  - Line 2231, Column 28
  - ... and 4 more occurrences

- **merged_metadata** (appears 4 times):
  - Line 3197, Column 15
  - Line 3193, Column 12
  - Line 3195, Column 12
  - Line 3196, Column 8

- **msg** (appears 2 times):
  - Line 1376, Column 25
  - Line 1376, Column 57

- **nested** (appears 1 times):
  - Line 2894, Column 36

- **nested_value** (appears 3 times):
  - Line 3299, Column 48
  - Line 2869, Column 39
  - Line 2873, Column 39

- **pipeline_overrides** (appears 6 times):
  - Line 2978, Column 8
  - Line 761, Column 62
  - Line 761, Column 28
  - Line 2976, Column 31
  - Line 2992, Column 21
  - ... and 1 more occurrences

- **r** (appears 10 times):
  - Line 4891, Column 45
  - Line 4892, Column 41
  - Line 4893, Column 40
  - Line 4896, Column 28
  - Line 4898, Column 31
  - ... and 5 more occurrences

- **sequence_only** (appears 1 times):
  - Line 1047, Column 11

- **sources** (appears 1 times):
  - Line 807, Column 22

- **step_failures** (appears 2 times):
  - Line 1230, Column 12
  - Line 2210, Column 34

- **step_index** (appears 1 times):
  - Line 1158, Column 27

- **sub_pipeline_name** (appears 3 times):
  - Line 4862, Column 46
  - Line 4866, Column 46
  - Line 4868, Column 58

- **summary_parts** (appears 3 times):
  - Line 1384, Column 26
  - Line 1371, Column 8
  - Line 1382, Column 12

- **total_steps** (appears 1 times):
  - Line 1158, Column 40

- **training_input** (appears 3 times):
  - Line 3165, Column 15
  - Line 3163, Column 12
  - Line 2938, Column 54

- **v** (appears 8 times):
  - Line 3045, Column 34
  - Line 1429, Column 38
  - Line 1575, Column 51
  - Line 3353, Column 47
  - Line 1429, Column 71
  - ... and 3 more occurrences

- **values** (appears 5 times):
  - Line 2900, Column 15
  - Line 2903, Column 20
  - Line 2907, Column 25
  - Line 2905, Column 53
  - Line 2896, Column 16

- **visited_frames** (appears 3 times):
  - Line 3219, Column 24
  - Line 3277, Column 12
  - Line 3220, Column 40

- **warnings** (appears 2 times):
  - Line 1103, Column 15
  - Line 1102, Column 30

---

### 14. src/feature_generation/utils/vectorbt_rolling_optimizer.py

**Total Issues:** 252

**Issues by Name:**

- **args** (appears 8 times):
  - Line 33, Column 51
  - Line 34, Column 56
  - Line 35, Column 54
  - Line 36, Column 60
  - Line 37, Column 56
  - ... and 3 more occurrences

- **attr_name** (appears 2 times):
  - Line 1645, Column 15
  - Line 1646, Column 30

- **column** (appears 28 times):
  - Line 1360, Column 58
  - Line 1361, Column 32
  - Line 1364, Column 37
  - Line 1375, Column 30
  - Line 1379, Column 33
  - ... and 23 more occurrences

- **e** (appears 72 times):
  - Line 199, Column 66
  - Line 412, Column 56
  - Line 563, Column 71
  - Line 589, Column 70
  - Line 1091, Column 57
  - ... and 67 more occurrences

- **fallback_error** (appears 2 times):
  - Line 711, Column 173
  - Line 710, Column 53

- **i** (appears 6 times):
  - Line 741, Column 15
  - Line 768, Column 15
  - Line 728, Column 30
  - Line 755, Column 30
  - Line 728, Column 32
  - ... and 1 more occurrences

- **k** (appears 2 times):
  - Line 1613, Column 67
  - Line 1519, Column 78

- **key** (appears 3 times):
  - Line 1521, Column 46
  - Line 1522, Column 52
  - Line 1523, Column 53

- **kwargs** (appears 127 times):
  - Line 1693, Column 50
  - Line 1699, Column 49
  - Line 1705, Column 49
  - Line 1711, Column 49
  - Line 1717, Column 49
  - ... and 122 more occurrences

- **x** (appears 2 times):
  - Line 358, Column 33
  - Line 404, Column 33

---

### 15. src/training/steps/pre_training/multi_horizon_profit_labeler.py

**Total Issues:** 252

**Issues by Name:**

- **alias** (appears 1 times):
  - Line 2254, Column 31

- **args** (appears 3 times):
  - Line 106, Column 24
  - Line 114, Column 46
  - Line 117, Column 46

- **base** (appears 15 times):
  - Line 2100, Column 67
  - Line 2104, Column 63
  - Line 2121, Column 68
  - Line 2125, Column 64
  - Line 2140, Column 66
  - ... and 10 more occurrences

- **base_dir** (appears 2 times):
  - Line 262, Column 11
  - Line 260, Column 4

- **batch** (appears 3 times):
  - Line 914, Column 34
  - Line 898, Column 47
  - Line 1640, Column 30

- **batch_size** (appears 9 times):
  - Line 1801, Column 41
  - Line 1723, Column 74
  - Line 1797, Column 11
  - Line 1797, Column 33
  - Line 1797, Column 65
  - ... and 4 more occurrences

- **candidate** (appears 2 times):
  - Line 1894, Column 19
  - Line 1895, Column 42

- **chunk** (appears 3 times):
  - Line 1724, Column 18
  - Line 1785, Column 22
  - Line 1682, Column 26

- **col** (appears 36 times):
  - Line 1883, Column 30
  - Line 1899, Column 19
  - Line 1425, Column 28
  - Line 1461, Column 31
  - Line 1907, Column 15
  - ... and 31 more occurrences

- **column** (appears 2 times):
  - Line 321, Column 15
  - Line 322, Column 22

- **contract_error** (appears 1 times):
  - Line 1121, Column 68

- **correlation_id** (appears 1 times):
  - Line 278, Column 23

- **data_locator** (appears 8 times):
  - Line 2769, Column 15
  - Line 2774, Column 15
  - Line 2770, Column 51
  - Line 2771, Column 53
  - Line 2779, Column 32
  - ... and 3 more occurrences

- **data_type** (appears 10 times):
  - Line 1740, Column 65
  - Line 1668, Column 77
  - Line 1677, Column 20
  - Line 1691, Column 16
  - Line 1703, Column 77
  - ... and 5 more occurrences

- **e** (appears 27 times):
  - Line 2676, Column 29
  - Line 2873, Column 42
  - Line 1166, Column 63
  - Line 1208, Column 33
  - Line 1387, Column 82
  - ... and 22 more occurrences

- **expected_name** (appears 1 times):
  - Line 2162, Column 39

- **feature_frames** (appears 5 times):
  - Line 1339, Column 15
  - Line 1009, Column 16
  - Line 1018, Column 49
  - Line 1018, Column 31
  - Line 1340, Column 35

- **feature_metadata** (appears 7 times):
  - Line 1323, Column 15
  - Line 1010, Column 16
  - Line 1019, Column 53
  - Line 1019, Column 33
  - Line 1324, Column 38
  - ... and 2 more occurrences

- **fold** (appears 5 times):
  - Line 831, Column 22
  - Line 834, Column 16
  - Line 2408, Column 22
  - Line 825, Column 13
  - Line 2417, Column 28

- **handle** (appears 1 times):
  - Line 270, Column 8

- **horizon** (appears 2 times):
  - Line 2280, Column 31
  - Line 2281, Column 28

- **info_error** (appears 1 times):
  - Line 1742, Column 84

- **item** (appears 1 times):
  - Line 187, Column 102

- **k** (appears 3 times):
  - Line 187, Column 20
  - Line 2286, Column 27
  - Line 1520, Column 28

- **key** (appears 5 times):
  - Line 2882, Column 16
  - Line 2708, Column 19
  - Line 2710, Column 38
  - Line 2711, Column 39
  - Line 2884, Column 19

- **kwargs** (appears 15 times):
  - Line 86, Column 26
  - Line 107, Column 26
  - Line 114, Column 54
  - Line 117, Column 54
  - Line 92, Column 26
  - ... and 10 more occurrences

- **load_error** (appears 2 times):
  - Line 1695, Column 89
  - Line 1762, Column 110

- **load_errors** (appears 9 times):
  - Line 1647, Column 11
  - Line 1648, Column 23
  - Line 1680, Column 20
  - Line 1706, Column 12
  - Line 1698, Column 12
  - ... and 4 more occurrences

- **logger** (appears 2 times):
  - Line 295, Column 8
  - Line 297, Column 8

- **mapping_metrics** (appears 4 times):
  - Line 1150, Column 15
  - Line 971, Column 32
  - Line 1153, Column 20
  - Line 1151, Column 75

- **market_data_batches** (appears 5 times):
  - Line 900, Column 23
  - Line 1087, Column 49
  - Line 898, Column 20
  - Line 904, Column 40
  - Line 932, Column 27

- **msg** (appears 1 times):
  - Line 1649, Column 34

- **params** (appears 2 times):
  - Line 1310, Column 34
  - Line 1308, Column 24

- **prep_error** (appears 2 times):
  - Line 1713, Column 93
  - Line 1777, Column 110

- **quality_score** (appears 7 times):
  - Line 1972, Column 53
  - Line 2661, Column 43
  - Line 2662, Column 42
  - Line 2663, Column 37
  - Line 2664, Column 35
  - ... and 2 more occurrences

- **quality_thresholds** (appears 2 times):
  - Line 873, Column 25
  - Line 874, Column 15

- **regime** (appears 10 times):
  - Line 1954, Column 52
  - Line 1966, Column 34
  - Line 1951, Column 57
  - Line 1968, Column 45
  - Line 1970, Column 53
  - ... and 5 more occurrences

- **regime_execution_timing** (appears 3 times):
  - Line 1999, Column 19
  - Line 1976, Column 24
  - Line 2001, Column 79

- **register_error** (appears 1 times):
  - Line 2866, Column 78

- **scaling_error** (appears 2 times):
  - Line 1291, Column 42
  - Line 1293, Column 82

- **shift** (appears 2 times):
  - Line 1305, Column 39
  - Line 1305, Column 85

- **source_columns** (appears 2 times):
  - Line 2158, Column 19
  - Line 2160, Column 33

- **summaries** (appears 2 times):
  - Line 2429, Column 15
  - Line 2415, Column 12

- **sym** (appears 1 times):
  - Line 1632, Column 24

- **target** (appears 6 times):
  - Line 2302, Column 40
  - Line 2303, Column 47
  - Line 2357, Column 19
  - Line 2358, Column 29
  - Line 2359, Column 42
  - ... and 1 more occurrences

- **target_name** (appears 1 times):
  - Line 2660, Column 36

- **tf** (appears 1 times):
  - Line 1633, Column 24

- **token** (appears 1 times):
  - Line 2369, Column 23

- **v** (appears 6 times):
  - Line 187, Column 44
  - Line 189, Column 36
  - Line 2286, Column 30
  - Line 1520, Column 54
  - Line 1522, Column 51
  - ... and 1 more occurrences

- **validation_error** (appears 5 times):
  - Line 2914, Column 32
  - Line 2923, Column 38
  - Line 2924, Column 35
  - Line 2925, Column 92
  - Line 2925, Column 59

- **window_days** (appears 4 times):
  - Line 1672, Column 15
  - Line 1678, Column 20
  - Line 1636, Column 36
  - Line 1750, Column 64

---

### 16. src/utils/feature_output_validator.py

**Total Issues:** 251

**Issues by Name:**

- **aligned_series** (appears 3 times):
  - Line 262, Column 41
  - Line 259, Column 32
  - Line 261, Column 32

- **col** (appears 53 times):
  - Line 328, Column 58
  - Line 346, Column 33
  - Line 465, Column 33
  - Line 474, Column 33
  - Line 487, Column 33
  - ... and 48 more occurrences

- **col1** (appears 2 times):
  - Line 409, Column 35
  - Line 410, Column 55

- **col2** (appears 2 times):
  - Line 409, Column 60
  - Line 410, Column 66

- **critical** (appears 1 times):
  - Line 205, Column 36

- **duplicate_features** (appears 3 times):
  - Line 411, Column 15
  - Line 412, Column 75
  - Line 410, Column 24

- **e** (appears 1 times):
  - Line 282, Column 114

- **extreme_features** (appears 3 times):
  - Line 479, Column 11
  - Line 478, Column 16
  - Line 480, Column 91

- **feature_type** (appears 1 times):
  - Line 337, Column 15

- **high_cardinality_features** (appears 3 times):
  - Line 498, Column 11
  - Line 497, Column 16
  - Line 499, Column 78

- **i** (appears 9 times):
  - Line 503, Column 27
  - Line 398, Column 31
  - Line 408, Column 70
  - Line 399, Column 54
  - Line 408, Column 61
  - ... and 4 more occurrences

- **invalid_names** (appears 3 times):
  - Line 304, Column 15
  - Line 303, Column 20
  - Line 305, Column 78

- **issue** (appears 1 times):
  - Line 205, Column 45

- **j** (appears 5 times):
  - Line 399, Column 57
  - Line 504, Column 45
  - Line 505, Column 85
  - Line 401, Column 89
  - Line 506, Column 85

- **key** (appears 1 times):
  - Line 266, Column 91

- **keyword** (appears 5 times):
  - Line 565, Column 70
  - Line 538, Column 81
  - Line 541, Column 83
  - Line 545, Column 91
  - Line 549, Column 84

- **low_variance_features** (appears 3 times):
  - Line 524, Column 11
  - Line 523, Column 16
  - Line 525, Column 85

- **name** (appears 10 times):
  - Line 224, Column 64
  - Line 273, Column 63
  - Line 246, Column 63
  - Line 239, Column 78
  - Line 227, Column 76
  - ... and 5 more occurrences

- **non_numeric_features** (appears 3 times):
  - Line 320, Column 11
  - Line 319, Column 16
  - Line 321, Column 73

- **object_features** (appears 3 times):
  - Line 326, Column 11
  - Line 325, Column 16
  - Line 327, Column 74

- **perfect_correlations** (appears 6 times):
  - Line 507, Column 11
  - Line 403, Column 15
  - Line 506, Column 20
  - Line 508, Column 85
  - Line 402, Column 24
  - ... and 1 more occurrences

- **recommendations** (appears 11 times):
  - Line 601, Column 15
  - Line 573, Column 12
  - Line 575, Column 12
  - Line 582, Column 12
  - Line 585, Column 12
  - ... and 6 more occurrences

- **scaler_problematic** (appears 4 times):
  - Line 470, Column 11
  - Line 467, Column 16
  - Line 469, Column 16
  - Line 471, Column 85

- **sklearn_incompatible** (appears 4 times):
  - Line 461, Column 11
  - Line 458, Column 16
  - Line 460, Column 16
  - Line 462, Column 78

- **sparse_features** (appears 3 times):
  - Line 531, Column 11
  - Line 530, Column 16
  - Line 532, Column 92

- **thresholds** (appears 1 times):
  - Line 338, Column 23

- **v** (appears 4 times):
  - Line 220, Column 35
  - Line 230, Column 35
  - Line 232, Column 35
  - Line 263, Column 35

- **validation_results** (appears 61 times):
  - Line 206, Column 15
  - Line 90, Column 19
  - Line 99, Column 71
  - Line 115, Column 61
  - Line 131, Column 78
  - ... and 56 more occurrences

- **value** (appears 24 times):
  - Line 235, Column 27
  - Line 237, Column 38
  - Line 267, Column 35
  - Line 270, Column 27
  - Line 272, Column 38
  - ... and 19 more occurrences

- **w** (appears 16 times):
  - Line 565, Column 31
  - Line 580, Column 28
  - Line 583, Column 28
  - Line 586, Column 26
  - Line 589, Column 30
  - ... and 11 more occurrences

- **zero_var_features** (appears 2 times):
  - Line 490, Column 34
  - Line 489, Column 16

---

### 17. src/training/steps/model_training/tactician_ensemble_training.py

**Total Issues:** 246

**Issues by Name:**

- **args** (appears 1 times):
  - Line 2360, Column 141

- **attr_name** (appears 3 times):
  - Line 2194, Column 33
  - Line 2196, Column 49
  - Line 2202, Column 38

- **base_model** (appears 1 times):
  - Line 1375, Column 67

- **base_name** (appears 5 times):
  - Line 1545, Column 94
  - Line 1375, Column 95
  - Line 1379, Column 108
  - Line 1381, Column 88
  - Line 1377, Column 75

- **catboost_error** (appears 1 times):
  - Line 1914, Column 81

- **cleanup_e** (appears 1 times):
  - Line 1570, Column 73

- **cleanup_error** (appears 2 times):
  - Line 2182, Column 73
  - Line 2183, Column 74

- **col** (appears 2 times):
  - Line 2487, Column 34
  - Line 2487, Column 69

- **conv_error** (appears 1 times):
  - Line 1749, Column 107

- **drop_idx** (appears 1 times):
  - Line 1063, Column 50

- **e** (appears 84 times):
  - Line 138, Column 83
  - Line 177, Column 105
  - Line 233, Column 94
  - Line 244, Column 103
  - Line 268, Column 105
  - ... and 79 more occurrences

- **ensemble** (appears 5 times):
  - Line 2497, Column 23
  - Line 1390, Column 69
  - Line 2497, Column 44
  - Line 2500, Column 80
  - Line 2500, Column 41

- **ensemble_name** (appears 12 times):
  - Line 1390, Column 82
  - Line 1524, Column 90
  - Line 1392, Column 57
  - Line 1395, Column 100
  - Line 1397, Column 117
  - ... and 7 more occurrences

- **error** (appears 1 times):
  - Line 599, Column 40

- **fallback_error** (appears 1 times):
  - Line 2493, Column 83

- **feature** (appears 10 times):
  - Line 2400, Column 19
  - Line 2449, Column 23
  - Line 2401, Column 37
  - Line 2401, Column 62
  - Line 2402, Column 37
  - ... and 5 more occurrences

- **feature_name** (appears 2 times):
  - Line 2467, Column 38
  - Line 2480, Column 38

- **feature_value** (appears 2 times):
  - Line 2467, Column 54
  - Line 2480, Column 54

- **fold_idx** (appears 4 times):
  - Line 1665, Column 43
  - Line 1677, Column 72
  - Line 1682, Column 80
  - Line 1689, Column 87

- **kwargs** (appears 19 times):
  - Line 2360, Column 149
  - Line 2653, Column 77
  - Line 2696, Column 61
  - Line 2699, Column 62
  - Line 2657, Column 59
  - ... and 14 more occurrences

- **lgbm_error** (appears 1 times):
  - Line 1902, Column 81

- **m** (appears 3 times):
  - Line 2146, Column 30
  - Line 2146, Column 88
  - Line 2146, Column 94

- **meta_learner_name** (appears 1 times):
  - Line 1995, Column 52

- **metric** (appears 1 times):
  - Line 2273, Column 41

- **metric_name** (appears 4 times):
  - Line 2146, Column 36
  - Line 2148, Column 55
  - Line 2146, Column 73
  - Line 2146, Column 96

- **model_data** (appears 12 times):
  - Line 2029, Column 42
  - Line 2054, Column 70
  - Line 2029, Column 57
  - Line 2042, Column 49
  - Line 2043, Column 54
  - ... and 7 more occurrences

- **model_error** (appears 2 times):
  - Line 2509, Column 100
  - Line 2530, Column 101

- **name** (appears 2 times):
  - Line 1130, Column 29
  - Line 1132, Column 58

- **nas_model** (appears 4 times):
  - Line 2519, Column 23
  - Line 2519, Column 45
  - Line 2521, Column 81
  - Line 2521, Column 41

- **p** (appears 4 times):
  - Line 2076, Column 44
  - Line 2077, Column 41
  - Line 2076, Column 80
  - Line 2077, Column 81

- **pred_error** (appears 1 times):
  - Line 1736, Column 82

- **r** (appears 4 times):
  - Line 2099, Column 47
  - Line 2100, Column 43
  - Line 2099, Column 111
  - Line 2100, Column 103

- **regime** (appears 8 times):
  - Line 2136, Column 46
  - Line 1998, Column 43
  - Line 2503, Column 57
  - Line 2524, Column 52
  - Line 2509, Column 90
  - ... and 3 more occurrences

- **regime_id** (appears 5 times):
  - Line 2040, Column 45
  - Line 2050, Column 45
  - Line 2031, Column 45
  - Line 2039, Column 48
  - Line 2049, Column 62

- **regime_metrics** (appears 3 times):
  - Line 1987, Column 30
  - Line 1987, Column 71
  - Line 1991, Column 54

- **regime_models** (appears 2 times):
  - Line 2027, Column 30
  - Line 2028, Column 50

- **resource_error** (appears 3 times):
  - Line 2204, Column 90
  - Line 2205, Column 82
  - Line 2215, Column 90

- **resource_name** (appears 6 times):
  - Line 2210, Column 33
  - Line 2212, Column 38
  - Line 2199, Column 60
  - Line 2204, Column 58
  - Line 2205, Column 65
  - ... and 1 more occurrences

- **te** (appears 1 times):
  - Line 1381, Column 101

- **tr_idx** (appears 1 times):
  - Line 1665, Column 80

- **utility** (appears 8 times):
  - Line 587, Column 44
  - Line 589, Column 47
  - Line 2280, Column 48
  - Line 591, Column 42
  - Line 593, Column 44
  - ... and 3 more occurrences

- **v** (appears 1 times):
  - Line 436, Column 60

- **va_idx** (appears 6 times):
  - Line 1686, Column 27
  - Line 1671, Column 34
  - Line 1676, Column 48
  - Line 1685, Column 24
  - Line 1669, Column 39
  - ... and 1 more occurrences

- **x** (appears 4 times):
  - Line 2433, Column 96
  - Line 2433, Column 109
  - Line 2438, Column 98
  - Line 2438, Column 111

- **xgb_error** (appears 1 times):
  - Line 1889, Column 80

---

### 18. src/training/steps/pre_training/unified_data_driven_pipeline/consolidated_pipeline_runner.py

**Total Issues:** 244

**Issues by Name:**

- **args** (appears 10 times):
  - Line 37, Column 56
  - Line 25, Column 51
  - Line 26, Column 54
  - Line 27, Column 60
  - Line 28, Column 60
  - ... and 5 more occurrences

- **data** (appears 60 times):
  - Line 1357, Column 53
  - Line 1381, Column 56
  - Line 1405, Column 55
  - Line 1429, Column 57
  - Line 1453, Column 59
  - ... and 55 more occurrences

- **direction** (appears 11 times):
  - Line 165, Column 29
  - Line 264, Column 29
  - Line 322, Column 29
  - Line 386, Column 29
  - Line 448, Column 29
  - ... and 6 more occurrences

- **e** (appears 54 times):
  - Line 1361, Column 43
  - Line 1385, Column 43
  - Line 1409, Column 43
  - Line 1433, Column 43
  - Line 1457, Column 43
  - ... and 49 more occurrences

- **end_date** (appears 10 times):
  - Line 168, Column 28
  - Line 267, Column 28
  - Line 325, Column 28
  - Line 389, Column 28
  - Line 451, Column 28
  - ... and 5 more occurrences

- **exchange** (appears 11 times):
  - Line 169, Column 28
  - Line 268, Column 28
  - Line 326, Column 28
  - Line 390, Column 28
  - Line 452, Column 28
  - ... and 6 more occurrences

- **f** (appears 9 times):
  - Line 1036, Column 12
  - Line 1078, Column 12
  - Line 1120, Column 12
  - Line 1162, Column 12
  - Line 1204, Column 12
  - ... and 4 more occurrences

- **key** (appears 5 times):
  - Line 882, Column 39
  - Line 883, Column 52
  - Line 884, Column 40
  - Line 885, Column 44
  - Line 887, Column 66

- **kwargs** (appears 20 times):
  - Line 37, Column 64
  - Line 25, Column 59
  - Line 26, Column 62
  - Line 27, Column 68
  - Line 28, Column 68
  - ... and 15 more occurrences

- **lookback_days** (appears 10 times):
  - Line 166, Column 33
  - Line 265, Column 33
  - Line 323, Column 33
  - Line 387, Column 33
  - Line 449, Column 33
  - ... and 5 more occurrences

- **start_date** (appears 10 times):
  - Line 167, Column 30
  - Line 266, Column 30
  - Line 324, Column 30
  - Line 388, Column 30
  - Line 450, Column 30
  - ... and 5 more occurrences

- **symbol** (appears 11 times):
  - Line 163, Column 26
  - Line 262, Column 26
  - Line 320, Column 26
  - Line 384, Column 26
  - Line 446, Column 26
  - ... and 6 more occurrences

- **timeframe** (appears 21 times):
  - Line 164, Column 29
  - Line 263, Column 29
  - Line 321, Column 29
  - Line 385, Column 29
  - Line 447, Column 29
  - ... and 16 more occurrences

- **value** (appears 2 times):
  - Line 884, Column 45
  - Line 885, Column 67

---

### 19. src/training/steps/model_training/__init__.py

**Total Issues:** 244

**Issues by Name:**

- **AnalystCreationStep** (appears 1 times):
  - Line 983, Column 37

- **AnalystModelTrainer** (appears 1 times):
  - Line 975, Column 43

- **HMMBasedTrainingStep** (appears 1 times):
  - Line 981, Column 39

- **TacticianModelTrainer** (appears 1 times):
  - Line 976, Column 45

- **UnifiedRegimeIntelligenceStep** (appears 1 times):
  - Line 982, Column 48

- **_execute_training_step** (appears 1 times):
  - Line 998, Column 33

- **_extract_models_and_data** (appears 1 times):
  - Line 873, Column 84

- **_load_model_specific_data** (appears 1 times):
  - Line 899, Column 53

- **_monitor_memory_usage** (appears 4 times):
  - Line 939, Column 31
  - Line 968, Column 33
  - Line 1031, Column 33
  - Line 1010, Column 40

- **_run_model_interpretability_analysis** (appears 1 times):
  - Line 488, Column 53

- **_validate_data_quality** (appears 1 times):
  - Line 960, Column 35

- **_validate_pipeline_inputs** (appears 1 times):
  - Line 942, Column 29

- **_validate_step_dependencies** (appears 1 times):
  - Line 951, Column 35

- **analyst** (appears 4 times):
  - Line 656, Column 35
  - Line 657, Column 56
  - Line 658, Column 37
  - Line 659, Column 56

- **cls** (appears 2 times):
  - Line 988, Column 32
  - Line 1017, Column 32

- **col** (appears 14 times):
  - Line 76, Column 31
  - Line 408, Column 19
  - Line 76, Column 66
  - Line 828, Column 42
  - Line 410, Column 68
  - ... and 9 more occurrences

- **config** (appears 17 times):
  - Line 281, Column 26
  - Line 461, Column 39
  - Line 1032, Column 231
  - Line 282, Column 16
  - Line 937, Column 47
  - ... and 12 more occurrences

- **data_dir** (appears 29 times):
  - Line 251, Column 65
  - Line 255, Column 32
  - Line 313, Column 106
  - Line 942, Column 84
  - Line 951, Column 92
  - ... and 24 more occurrences

- **e** (appears 30 times):
  - Line 25, Column 60
  - Line 447, Column 61
  - Line 1062, Column 29
  - Line 1071, Column 81
  - Line 237, Column 63
  - ... and 25 more occurrences

- **enabled** (appears 4 times):
  - Line 988, Column 37
  - Line 988, Column 90
  - Line 1017, Column 37
  - Line 1017, Column 94

- **ensemble** (appears 4 times):
  - Line 681, Column 35
  - Line 682, Column 57
  - Line 683, Column 37
  - Line 684, Column 57

- **exchange** (appears 29 times):
  - Line 251, Column 29
  - Line 313, Column 60
  - Line 942, Column 63
  - Line 951, Column 71
  - Line 960, Column 66
  - ... and 24 more occurrences

- **file_name** (appears 3 times):
  - Line 261, Column 38
  - Line 270, Column 58
  - Line 275, Column 74

- **issue** (appears 4 times):
  - Line 291, Column 39
  - Line 292, Column 33
  - Line 434, Column 43
  - Line 435, Column 37

- **key** (appears 3 times):
  - Line 281, Column 15
  - Line 282, Column 23
  - Line 283, Column 82

- **keyword** (appears 1 times):
  - Line 515, Column 15

- **name** (appears 12 times):
  - Line 988, Column 26
  - Line 1017, Column 26
  - Line 706, Column 47
  - Line 728, Column 41
  - Line 750, Column 42
  - ... and 7 more occurrences

- **pattern** (appears 1 times):
  - Line 519, Column 15

- **step** (appears 10 times):
  - Line 1032, Column 259
  - Line 1038, Column 52
  - Line 319, Column 77
  - Line 1032, Column 297
  - Line 1032, Column 391
  - ... and 5 more occurrences

- **step_class** (appears 2 times):
  - Line 461, Column 28
  - Line 998, Column 67

- **step_index** (appears 5 times):
  - Line 995, Column 34
  - Line 996, Column 39
  - Line 1008, Column 19
  - Line 1015, Column 56
  - Line 1009, Column 64

- **symbol** (appears 30 times):
  - Line 251, Column 15
  - Line 313, Column 40
  - Line 942, Column 55
  - Line 951, Column 63
  - Line 960, Column 58
  - ... and 25 more occurrences

- **tactician** (appears 4 times):
  - Line 631, Column 35
  - Line 632, Column 58
  - Line 633, Column 37
  - Line 634, Column 58

- **timeframe** (appears 16 times):
  - Line 251, Column 46
  - Line 313, Column 83
  - Line 942, Column 73
  - Line 951, Column 81
  - Line 935, Column 78
  - ... and 11 more occurrences

- **warning** (appears 4 times):
  - Line 298, Column 41
  - Line 299, Column 33
  - Line 441, Column 45
  - Line 442, Column 37

---

### 20. src/training/steps/market_analysis/components/regime_models_training.py

**Total Issues:** 239

**Issues by Name:**

- **artifact_key** (appears 4 times):
  - Line 321, Column 88
  - Line 325, Column 88
  - Line 329, Column 84
  - Line 333, Column 92

- **artifact_value** (appears 9 times):
  - Line 318, Column 38
  - Line 319, Column 52
  - Line 320, Column 48
  - Line 323, Column 54
  - Line 324, Column 48
  - ... and 4 more occurrences

- **assignment_key** (appears 6 times):
  - Line 433, Column 31
  - Line 434, Column 74
  - Line 411, Column 39
  - Line 412, Column 72
  - Line 435, Column 112
  - ... and 1 more occurrences

- **base_name** (appears 10 times):
  - Line 1649, Column 38
  - Line 1652, Column 23
  - Line 1455, Column 123
  - Line 1457, Column 27
  - Line 1652, Column 54
  - ... and 5 more occurrences

- **category** (appears 5 times):
  - Line 1177, Column 69
  - Line 1174, Column 58
  - Line 1180, Column 77
  - Line 1207, Column 51
  - Line 1193, Column 46

- **class_weight** (appears 3 times):
  - Line 2412, Column 45
  - Line 2430, Column 52
  - Line 2422, Column 99

- **col** (appears 2 times):
  - Line 857, Column 31
  - Line 857, Column 66

- **e** (appears 79 times):
  - Line 2566, Column 18
  - Line 49, Column 45
  - Line 50, Column 65
  - Line 58, Column 41
  - Line 59, Column 61
  - ... and 74 more occurrences

- **f** (appears 1 times):
  - Line 2601, Column 49

- **fallback_error** (appears 1 times):
  - Line 1909, Column 85

- **fold** (appears 2 times):
  - Line 2082, Column 67
  - Line 2090, Column 67

- **generator** (appears 9 times):
  - Line 1189, Column 33
  - Line 1216, Column 33
  - Line 1188, Column 72
  - Line 1215, Column 88
  - Line 1219, Column 62
  - ... and 4 more occurrences

- **i** (appears 11 times):
  - Line 677, Column 55
  - Line 678, Column 55
  - Line 1851, Column 34
  - Line 680, Column 39
  - Line 1851, Column 56
  - ... and 6 more occurrences

- **idx** (appears 6 times):
  - Line 1836, Column 52
  - Line 1841, Column 41
  - Line 1856, Column 49
  - Line 1857, Column 56
  - Line 1900, Column 49
  - ... and 1 more occurrences

- **info** (appears 16 times):
  - Line 1921, Column 15
  - Line 1794, Column 19
  - Line 1838, Column 12
  - Line 1850, Column 16
  - Line 1862, Column 16
  - ... and 11 more occurrences

- **item** (appears 2 times):
  - Line 1864, Column 23
  - Line 1864, Column 42

- **k** (appears 2 times):
  - Line 465, Column 31
  - Line 510, Column 31

- **key** (appears 3 times):
  - Line 310, Column 23
  - Line 311, Column 50
  - Line 312, Column 78

- **m** (appears 3 times):
  - Line 1561, Column 64
  - Line 1561, Column 94
  - Line 1563, Column 74

- **max_depth** (appears 3 times):
  - Line 2410, Column 42
  - Line 2429, Column 49
  - Line 2422, Column 73

- **name** (appears 16 times):
  - Line 962, Column 39
  - Line 1442, Column 39
  - Line 972, Column 31
  - Line 1446, Column 31
  - Line 1450, Column 19
  - ... and 11 more occurrences

- **outcome_file** (appears 4 times):
  - Line 2600, Column 30
  - Line 2598, Column 74
  - Line 2606, Column 85
  - Line 2621, Column 81

- **p** (appears 2 times):
  - Line 702, Column 62
  - Line 702, Column 73

- **param_grid** (appears 3 times):
  - Line 2404, Column 33
  - Line 2405, Column 40
  - Line 2399, Column 71

- **params** (appears 3 times):
  - Line 2305, Column 100
  - Line 2311, Column 34
  - Line 2332, Column 38

- **rank** (appears 1 times):
  - Line 1858, Column 36

- **regime_data** (appears 8 times):
  - Line 759, Column 30
  - Line 768, Column 32
  - Line 761, Column 52
  - Line 762, Column 50
  - Line 763, Column 56
  - ... and 3 more occurrences

- **regime_key** (appears 1 times):
  - Line 760, Column 36

- **train_idx** (appears 4 times):
  - Line 2053, Column 33
  - Line 2052, Column 45
  - Line 1995, Column 41
  - Line 1994, Column 53

- **v** (appears 2 times):
  - Line 465, Column 39
  - Line 510, Column 39

- **val_idx** (appears 7 times):
  - Line 2052, Column 59
  - Line 2076, Column 30
  - Line 2086, Column 30
  - Line 2094, Column 30
  - Line 1994, Column 67
  - ... and 2 more occurrences

- **x** (appears 11 times):
  - Line 2593, Column 45
  - Line 2133, Column 48
  - Line 2195, Column 48
  - Line 2133, Column 38
  - Line 2195, Column 38
  - ... and 6 more occurrences

---

### 21. src/utils/ml_common/optimization/bayesian_tpe_optimizer.py

**Total Issues:** 238

**Issues by Name:**

- **constraint_func** (appears 1 times):
  - Line 2158, Column 31

- **e** (appears 34 times):
  - Line 36, Column 60
  - Line 53, Column 60
  - Line 306, Column 87
  - Line 371, Column 87
  - Line 494, Column 69
  - ... and 29 more occurrences

- **h** (appears 3 times):
  - Line 2292, Column 24
  - Line 2312, Column 31
  - Line 2316, Column 46

- **history** (appears 4 times):
  - Line 650, Column 28
  - Line 646, Column 19
  - Line 2302, Column 27
  - Line 608, Column 19

- **i** (appears 31 times):
  - Line 1506, Column 29
  - Line 2031, Column 29
  - Line 2304, Column 29
  - Line 658, Column 44
  - Line 998, Column 44
  - ... and 26 more occurrences

- **imp** (appears 2 times):
  - Line 1010, Column 26
  - Line 669, Column 30

- **j** (appears 4 times):
  - Line 1377, Column 61
  - Line 1855, Column 61
  - Line 1950, Column 61
  - Line 1782, Column 39

- **k** (appears 1 times):
  - Line 2306, Column 37

- **key** (appears 4 times):
  - Line 181, Column 40
  - Line 394, Column 40
  - Line 182, Column 41
  - Line 395, Column 41

- **kwargs** (appears 13 times):
  - Line 179, Column 11
  - Line 230, Column 41
  - Line 239, Column 39
  - Line 240, Column 33
  - Line 246, Column 49
  - ... and 8 more occurrences

- **m** (appears 3 times):
  - Line 2242, Column 44
  - Line 2243, Column 46
  - Line 2244, Column 48

- **metric** (appears 2 times):
  - Line 2234, Column 24
  - Line 2237, Column 48

- **metrics** (appears 5 times):
  - Line 2241, Column 19
  - Line 2250, Column 47
  - Line 2242, Column 67
  - Line 2243, Column 83
  - Line 2244, Column 83

- **name** (appears 21 times):
  - Line 1613, Column 24
  - Line 1676, Column 29
  - Line 1735, Column 29
  - Line 1770, Column 29
  - Line 2086, Column 29
  - ... and 16 more occurrences

- **obj_name** (appears 8 times):
  - Line 704, Column 32
  - Line 602, Column 15
  - Line 591, Column 48
  - Line 592, Column 27
  - Line 592, Column 56
  - ... and 3 more occurrences

- **param_config** (appears 61 times):
  - Line 1259, Column 45
  - Line 1266, Column 28
  - Line 1264, Column 26
  - Line 1273, Column 28
  - Line 1292, Column 45
  - ... and 56 more occurrences

- **param_list** (appears 6 times):
  - Line 1365, Column 48
  - Line 1843, Column 48
  - Line 1938, Column 48
  - Line 1376, Column 37
  - Line 1854, Column 37
  - ... and 1 more occurrences

- **point** (appears 1 times):
  - Line 911, Column 31

- **t** (appears 23 times):
  - Line 2206, Column 28
  - Line 2207, Column 25
  - Line 2208, Column 23
  - Line 2209, Column 27
  - Line 1220, Column 18
  - ... and 18 more occurrences

- **v** (appears 5 times):
  - Line 2306, Column 67
  - Line 2306, Column 86
  - Line 2306, Column 41
  - Line 2306, Column 47
  - Line 1722, Column 55

- **val** (appears 6 times):
  - Line 1365, Column 30
  - Line 1843, Column 30
  - Line 1938, Column 30
  - Line 1355, Column 56
  - Line 1833, Column 56
  - ... and 1 more occurrences

---

### 22. src/training/steps/model_training/analyst_models_training_refactored.py

**Total Issues:** 234

**Issues by Name:**

- **check** (appears 2 times):
  - Line 3269, Column 31
  - Line 3270, Column 30

- **ctx** (appears 1 times):
  - Line 852, Column 20

- **e** (appears 89 times):
  - Line 2230, Column 47
  - Line 124, Column 54
  - Line 295, Column 63
  - Line 327, Column 61
  - Line 419, Column 56
  - ... and 84 more occurrences

- **f** (appears 8 times):
  - Line 1734, Column 39
  - Line 215, Column 32
  - Line 231, Column 33
  - Line 335, Column 36
  - Line 343, Column 37
  - ... and 3 more occurrences

- **i** (appears 6 times):
  - Line 2589, Column 42
  - Line 2589, Column 60
  - Line 2497, Column 66
  - Line 2499, Column 59
  - Line 2485, Column 52
  - ... and 1 more occurrences

- **issue** (appears 1 times):
  - Line 3548, Column 79

- **kwargs** (appears 1 times):
  - Line 215, Column 37

- **m** (appears 4 times):
  - Line 1608, Column 35
  - Line 1608, Column 75
  - Line 3055, Column 38
  - Line 3055, Column 93

- **mean_val** (appears 1 times):
  - Line 2497, Column 45

- **metric** (appears 4 times):
  - Line 2898, Column 19
  - Line 2899, Column 23
  - Line 2901, Column 41
  - Line 2905, Column 68

- **metric_name** (appears 3 times):
  - Line 3055, Column 40
  - Line 3057, Column 50
  - Line 3055, Column 78

- **metrics_list** (appears 3 times):
  - Line 3052, Column 23
  - Line 3054, Column 43
  - Line 3055, Column 62

- **model_info** (appears 61 times):
  - Line 1987, Column 38
  - Line 1992, Column 45
  - Line 1994, Column 40
  - Line 1996, Column 44
  - Line 1998, Column 46
  - ... and 56 more occurrences

- **model_name** (appears 6 times):
  - Line 2050, Column 61
  - Line 2087, Column 55
  - Line 2089, Column 53
  - Line 2091, Column 54
  - Line 3120, Column 50
  - ... and 1 more occurrences

- **model_results** (appears 12 times):
  - Line 1802, Column 42
  - Line 1802, Column 80
  - Line 2826, Column 42
  - Line 2826, Column 80
  - Line 3045, Column 42
  - ... and 7 more occurrences

- **mt** (appears 4 times):
  - Line 1544, Column 33
  - Line 1544, Column 68
  - Line 2333, Column 65
  - Line 2334, Column 40

- **outcome_error** (appears 1 times):
  - Line 2197, Column 74

- **p** (appears 3 times):
  - Line 3176, Column 27
  - Line 3176, Column 39
  - Line 3176, Column 68

- **regime_metrics** (appears 3 times):
  - Line 3110, Column 34
  - Line 3110, Column 75
  - Line 3115, Column 51

- **regime_results** (appears 8 times):
  - Line 1799, Column 34
  - Line 2822, Column 34
  - Line 3043, Column 34
  - Line 3203, Column 34
  - Line 1801, Column 57
  - ... and 3 more occurrences

- **report_error** (appears 1 times):
  - Line 2227, Column 69

- **row** (appears 1 times):
  - Line 2589, Column 32

- **warning** (appears 4 times):
  - Line 1649, Column 42
  - Line 2535, Column 42
  - Line 3258, Column 80
  - Line 1921, Column 53

- **weight** (appears 3 times):
  - Line 2907, Column 36
  - Line 2901, Column 51
  - Line 2906, Column 50

- **x** (appears 4 times):
  - Line 1192, Column 104
  - Line 3524, Column 75
  - Line 2107, Column 96
  - Line 2108, Column 97

---

### 23. src/training/steps/model_training/tactician_models_training_refactored.py

**Total Issues:** 232

**Issues by Name:**

- **array** (appears 3 times):
  - Line 1243, Column 61
  - Line 1238, Column 19
  - Line 1239, Column 91

- **available** (appears 1 times):
  - Line 1206, Column 84

- **e** (appears 100 times):
  - Line 48, Column 83
  - Line 63, Column 105
  - Line 97, Column 94
  - Line 110, Column 103
  - Line 121, Column 92
  - ... and 95 more occurrences

- **f** (appears 1 times):
  - Line 1861, Column 48

- **feature_array** (appears 2 times):
  - Line 2244, Column 27
  - Line 2245, Column 104

- **i** (appears 7 times):
  - Line 2049, Column 67
  - Line 2092, Column 66
  - Line 2180, Column 73
  - Line 2223, Column 71
  - Line 2245, Column 57
  - ... and 2 more occurrences

- **key** (appears 2 times):
  - Line 730, Column 36
  - Line 731, Column 37

- **kwargs** (appears 1 times):
  - Line 729, Column 30

- **model_info** (appears 64 times):
  - Line 1635, Column 38
  - Line 1640, Column 45
  - Line 1642, Column 40
  - Line 1644, Column 44
  - Line 1646, Column 46
  - ... and 59 more occurrences

- **model_name** (appears 10 times):
  - Line 1700, Column 61
  - Line 1737, Column 55
  - Line 1739, Column 53
  - Line 1741, Column 54
  - Line 2118, Column 57
  - ... and 5 more occurrences

- **mt** (appears 2 times):
  - Line 2602, Column 65
  - Line 2603, Column 40

- **name** (appears 2 times):
  - Line 1243, Column 53
  - Line 1239, Column 35

- **outcome_error** (appears 1 times):
  - Line 1867, Column 74

- **phase_data** (appears 4 times):
  - Line 1781, Column 40
  - Line 1782, Column 39
  - Line 1783, Column 49
  - Line 1784, Column 38

- **recommendation** (appears 1 times):
  - Line 1344, Column 45

- **regime** (appears 14 times):
  - Line 2529, Column 56
  - Line 2595, Column 34
  - Line 2601, Column 23
  - Line 2540, Column 78
  - Line 3086, Column 44
  - ... and 9 more occurrences

- **regime_metrics** (appears 3 times):
  - Line 3075, Column 38
  - Line 3075, Column 79
  - Line 3079, Column 55

- **s** (appears 4 times):
  - Line 1756, Column 81
  - Line 1760, Column 83
  - Line 1762, Column 77
  - Line 1687, Column 57

- **utility** (appears 4 times):
  - Line 881, Column 44
  - Line 883, Column 47
  - Line 885, Column 42
  - Line 887, Column 44

- **warning** (appears 4 times):
  - Line 1027, Column 42
  - Line 1340, Column 50
  - Line 1503, Column 76
  - Line 2345, Column 62

- **x** (appears 2 times):
  - Line 1757, Column 96
  - Line 1758, Column 97

---

### 24. src/training/steps/model_training/tactician_pre_ml_orchestrator.py

**Total Issues:** 226

**Issues by Name:**

- **c** (appears 5 times):
  - Line 2493, Column 30
  - Line 2494, Column 29
  - Line 2493, Column 64
  - Line 2494, Column 63
  - Line 2493, Column 127

- **col** (appears 13 times):
  - Line 1107, Column 30
  - Line 1111, Column 34
  - Line 1279, Column 28
  - Line 1275, Column 31
  - Line 1278, Column 28
  - ... and 8 more occurrences

- **component_param** (appears 2 times):
  - Line 709, Column 30
  - Line 711, Column 30

- **config_attr** (appears 2 times):
  - Line 707, Column 41
  - Line 705, Column 40

- **custom_params** (appears 3 times):
  - Line 713, Column 45
  - Line 709, Column 16
  - Line 711, Column 16

- **e** (appears 55 times):
  - Line 40, Column 63
  - Line 61, Column 60
  - Line 69, Column 70
  - Line 79, Column 65
  - Line 93, Column 67
  - ... and 50 more occurrences

- **error** (appears 1 times):
  - Line 862, Column 42

- **f** (appears 9 times):
  - Line 1634, Column 34
  - Line 1634, Column 62
  - Line 2273, Column 37
  - Line 1777, Column 50
  - Line 1778, Column 54
  - ... and 4 more occurrences

- **feat** (appears 1 times):
  - Line 2280, Column 47

- **feature** (appears 2 times):
  - Line 1511, Column 34
  - Line 1477, Column 32

- **feature_name** (appears 8 times):
  - Line 2400, Column 19
  - Line 2401, Column 68
  - Line 2405, Column 36
  - Line 2415, Column 36
  - Line 2420, Column 36
  - ... and 3 more occurrences

- **gen_error** (appears 1 times):
  - Line 1666, Column 75

- **horizon** (appears 5 times):
  - Line 2433, Column 42
  - Line 2443, Column 42
  - Line 2445, Column 52
  - Line 2448, Column 42
  - Line 2450, Column 52

- **horizon_name** (appears 1 times):
  - Line 2160, Column 57

- **horizon_periods** (appears 1 times):
  - Line 2143, Column 64

- **i** (appears 1 times):
  - Line 2280, Column 42

- **k** (appears 2 times):
  - Line 2543, Column 42
  - Line 2557, Column 42

- **keyword** (appears 3 times):
  - Line 1282, Column 23
  - Line 1108, Column 35
  - Line 1289, Column 25

- **kw** (appears 5 times):
  - Line 1486, Column 25
  - Line 1488, Column 25
  - Line 1490, Column 25
  - Line 1492, Column 25
  - Line 1494, Column 25

- **label_error** (appears 1 times):
  - Line 2128, Column 74

- **long_lookbacks** (appears 1 times):
  - Line 1557, Column 66

- **long_pid_features** (appears 7 times):
  - Line 2037, Column 42
  - Line 2197, Column 42
  - Line 2332, Column 46
  - Line 2041, Column 51
  - Line 2201, Column 24
  - ... and 2 more occurrences

- **long_selected_features** (appears 3 times):
  - Line 2332, Column 85
  - Line 2337, Column 24
  - Line 2345, Column 218

- **long_signals** (appears 7 times):
  - Line 1382, Column 19
  - Line 1553, Column 19
  - Line 2037, Column 19
  - Line 2197, Column 19
  - Line 1386, Column 24
  - ... and 2 more occurrences

- **month** (appears 4 times):
  - Line 1975, Column 47
  - Line 1981, Column 47
  - Line 1986, Column 43
  - Line 1991, Column 43

- **opt_error** (appears 1 times):
  - Line 1458, Column 68

- **optimized_lookbacks** (appears 1 times):
  - Line 1650, Column 51

- **pid_features** (appears 4 times):
  - Line 2252, Column 33
  - Line 2259, Column 32
  - Line 2400, Column 35
  - Line 2401, Column 37

- **pnl** (appears 4 times):
  - Line 1975, Column 56
  - Line 1981, Column 56
  - Line 1986, Column 52
  - Line 1991, Column 52

- **score** (appears 1 times):
  - Line 2280, Column 55

- **short_lookbacks** (appears 1 times):
  - Line 1573, Column 67

- **short_pid_features** (appears 7 times):
  - Line 2052, Column 43
  - Line 2212, Column 43
  - Line 2348, Column 47
  - Line 2056, Column 52
  - Line 2216, Column 24
  - ... and 2 more occurrences

- **short_selected_features** (appears 3 times):
  - Line 2348, Column 88
  - Line 2353, Column 24
  - Line 2361, Column 222

- **short_signals** (appears 7 times):
  - Line 1397, Column 19
  - Line 1569, Column 19
  - Line 2052, Column 19
  - Line 2212, Column 19
  - Line 1401, Column 24
  - ... and 2 more occurrences

- **signal_type** (appears 33 times):
  - Line 2455, Column 41
  - Line 1629, Column 16
  - Line 1499, Column 19
  - Line 1433, Column 66
  - Line 1461, Column 80
  - ... and 28 more occurrences

- **status** (appears 3 times):
  - Line 692, Column 15
  - Line 686, Column 16
  - Line 688, Column 16

- **target_name** (appears 1 times):
  - Line 2160, Column 43

- **target_pct** (appears 2 times):
  - Line 2154, Column 48
  - Line 2165, Column 80

- **targets** (appears 2 times):
  - Line 2430, Column 42
  - Line 2452, Column 54

- **v** (appears 6 times):
  - Line 2543, Column 85
  - Line 2557, Column 85
  - Line 2543, Column 67
  - Line 2557, Column 67
  - Line 2543, Column 45
  - ... and 1 more occurrences

- **warning** (appears 1 times):
  - Line 858, Column 48

- **x** (appears 6 times):
  - Line 485, Column 108
  - Line 2268, Column 34
  - Line 1973, Column 101
  - Line 1979, Column 102
  - Line 1984, Column 97
  - ... and 1 more occurrences

---

### 25. exchanges/mexc.py

**Total Issues:** 222

**Issues by Name:**

- **client_order_id** (appears 3 times):
  - Line 590, Column 15
  - Line 575, Column 59
  - Line 591, Column 51

- **e** (appears 34 times):
  - Line 138, Column 61
  - Line 191, Column 61
  - Line 221, Column 53
  - Line 246, Column 58
  - Line 271, Column 45
  - ... and 29 more occurrences

- **end_time** (appears 1 times):
  - Line 522, Column 35

- **end_time_ms** (appears 2 times):
  - Line 754, Column 23
  - Line 795, Column 23

- **endpoint** (appears 1 times):
  - Line 670, Column 36

- **instrument** (appears 17 times):
  - Line 319, Column 19
  - Line 298, Column 27
  - Line 299, Column 36
  - Line 300, Column 39
  - Line 301, Column 30
  - ... and 12 more occurrences

- **item** (appears 58 times):
  - Line 806, Column 30
  - Line 722, Column 30
  - Line 722, Column 49
  - Line 726, Column 29
  - Line 727, Column 29
  - ... and 53 more occurrences

- **method** (appears 1 times):
  - Line 681, Column 44

- **order_id** (appears 6 times):
  - Line 603, Column 33
  - Line 614, Column 33
  - Line 888, Column 27
  - Line 899, Column 27
  - Line 892, Column 55
  - ... and 1 more occurrences

- **order_type** (appears 4 times):
  - Line 575, Column 30
  - Line 840, Column 20
  - Line 948, Column 62
  - Line 582, Column 24

- **position** (appears 2 times):
  - Line 869, Column 27
  - Line 868, Column 33

- **price** (appears 6 times):
  - Line 844, Column 11
  - Line 845, Column 36
  - Line 575, Column 52
  - Line 586, Column 15
  - Line 587, Column 44
  - ... and 1 more occurrences

- **quantity** (appears 4 times):
  - Line 841, Column 24
  - Line 575, Column 42
  - Line 583, Column 32
  - Line 948, Column 74

- **response** (appears 27 times):
  - Line 392, Column 19
  - Line 409, Column 19
  - Line 431, Column 19
  - Line 444, Column 19
  - Line 459, Column 19
  - ... and 22 more occurrences

- **side** (appears 4 times):
  - Line 575, Column 24
  - Line 839, Column 20
  - Line 948, Column 56
  - Line 581, Column 24

- **signed** (appears 1 times):
  - Line 676, Column 15

- **start_time** (appears 1 times):
  - Line 521, Column 37

- **start_time_ms** (appears 2 times):
  - Line 753, Column 25
  - Line 794, Column 25

- **stop_price** (appears 2 times):
  - Line 588, Column 15
  - Line 589, Column 48

- **symbol** (appears 46 times):
  - Line 626, Column 15
  - Line 857, Column 47
  - Line 876, Column 47
  - Line 575, Column 16
  - Line 700, Column 15
  - ... and 41 more occurrences

---

### 26. src/trading/reporting/daily_recorder.py

**Total Issues:** 216

**Issues by Name:**

- **days** (appears 2 times):
  - Line 751, Column 55
  - Line 721, Column 58

- **e** (appears 10 times):
  - Line 260, Column 67
  - Line 325, Column 64
  - Line 521, Column 65
  - Line 553, Column 70
  - Line 612, Column 65
  - ... and 5 more occurrences

- **existing_record** (appears 1 times):
  - Line 640, Column 36

- **f** (appears 4 times):
  - Line 635, Column 40
  - Line 659, Column 40
  - Line 682, Column 40
  - Line 252, Column 40

- **feature** (appears 7 times):
  - Line 492, Column 33
  - Line 492, Column 60
  - Line 484, Column 27
  - Line 487, Column 37
  - Line 488, Column 39
  - ... and 2 more occurrences

- **importance** (appears 1 times):
  - Line 487, Column 49

- **model_id** (appears 3 times):
  - Line 548, Column 37
  - Line 541, Column 71
  - Line 536, Column 53

- **p** (appears 10 times):
  - Line 344, Column 49
  - Line 345, Column 48
  - Line 346, Column 52
  - Line 350, Column 46
  - Line 351, Column 48
  - ... and 5 more occurrences

- **q** (appears 2 times):
  - Line 443, Column 57
  - Line 443, Column 91

- **r** (appears 2 times):
  - Line 621, Column 32
  - Line 621, Column 63

- **record** (appears 87 times):
  - Line 337, Column 12
  - Line 516, Column 12
  - Line 517, Column 12
  - Line 518, Column 12
  - Line 609, Column 12
  - ... and 82 more occurrences

- **regime** (appears 2 times):
  - Line 423, Column 38
  - Line 423, Column 66

- **row** (appears 1 times):
  - Line 661, Column 40

- **s** (appears 2 times):
  - Line 297, Column 16
  - Line 298, Column 19

- **session** (appears 3 times):
  - Line 507, Column 23
  - Line 508, Column 36
  - Line 508, Column 55

- **sessions** (appears 5 times):
  - Line 502, Column 15
  - Line 743, Column 61
  - Line 506, Column 31
  - Line 297, Column 27
  - Line 503, Column 44

- **t** (appears 25 times):
  - Line 291, Column 16
  - Line 536, Column 32
  - Line 341, Column 30
  - Line 370, Column 35
  - Line 375, Column 29
  - ... and 20 more occurrences

- **target_date** (appears 4 times):
  - Line 743, Column 71
  - Line 747, Column 50
  - Line 285, Column 26
  - Line 700, Column 41

- **trade** (appears 14 times):
  - Line 389, Column 46
  - Line 390, Column 44
  - Line 407, Column 23
  - Line 483, Column 47
  - Line 532, Column 34
  - ... and 9 more occurrences

- **trades** (appears 29 times):
  - Line 339, Column 15
  - Line 531, Column 25
  - Line 743, Column 53
  - Line 337, Column 38
  - Line 386, Column 29
  - ... and 24 more occurrences

- **x** (appears 2 times):
  - Line 495, Column 77
  - Line 424, Column 85

---

### 27. GUI/api_server.py

**Total Issues:** 216

**Issues by Name:**

- **_obs_exc** (appears 1 times):
  - Line 111, Column 71

- **a** (appears 13 times):
  - Line 1499, Column 25
  - Line 1500, Column 27
  - Line 1502, Column 28
  - Line 1503, Column 26
  - Line 1505, Column 34
  - ... and 8 more occurrences

- **b** (appears 2 times):
  - Line 546, Column 17
  - Line 546, Column 41

- **bot** (appears 1 times):
  - Line 925, Column 23

- **bot_id** (appears 2 times):
  - Line 941, Column 34
  - Line 950, Column 34

- **connection** (appears 1 times):
  - Line 381, Column 22

- **create_and_initialize** (appears 5 times):
  - Line 174, Column 32
  - Line 181, Column 30
  - Line 188, Column 32
  - Line 214, Column 23
  - Line 204, Column 38

- **days** (appears 3 times):
  - Line 530, Column 23
  - Line 819, Column 70
  - Line 822, Column 70

- **description** (appears 2 times):
  - Line 1727, Column 39
  - Line 1742, Column 39

- **e** (appears 46 times):
  - Line 56, Column 44
  - Line 1209, Column 47
  - Line 1236, Column 47
  - Line 1362, Column 47
  - Line 1419, Column 29
  - ... and 41 more occurrences

- **factory** (appears 1 times):
  - Line 142, Column 23

- **fi** (appears 1 times):
  - Line 1515, Column 16

- **file** (appears 5 times):
  - Line 1924, Column 55
  - Line 1923, Column 19
  - Line 1923, Column 48
  - Line 1927, Column 32
  - Line 1930, Column 45

- **get_ml_tracker_stats** (appears 3 times):
  - Line 1550, Column 18
  - Line 1555, Column 18
  - Line 1560, Column 18

- **get_model_performance** (appears 2 times):
  - Line 1370, Column 29
  - Line 1427, Column 29

- **i** (appears 10 times):
  - Line 453, Column 24
  - Line 531, Column 51
  - Line 648, Column 51
  - Line 812, Column 37
  - Line 1033, Column 56
  - ... and 5 more occurrences

- **init_and_start_dashboard** (appears 1 times):
  - Line 184, Column 8

- **init_and_start_ml** (appears 1 times):
  - Line 217, Column 8

- **init_only** (appears 2 times):
  - Line 177, Column 8
  - Line 191, Column 8

- **init_with_perf** (appears 1 times):
  - Line 207, Column 12

- **initializer** (appears 1 times):
  - Line 143, Column 27

- **inst** (appears 6 times):
  - Line 154, Column 21
  - Line 157, Column 19
  - Line 160, Column 14
  - Line 164, Column 21
  - Line 167, Column 19
  - ... and 1 more occurrences

- **limit** (appears 1 times):
  - Line 810, Column 23

- **metric** (appears 1 times):
  - Line 1078, Column 20

- **mode_name** (appears 5 times):
  - Line 1725, Column 54
  - Line 1726, Column 33
  - Line 1737, Column 62
  - Line 1741, Column 33
  - Line 1740, Column 68

- **model** (appears 15 times):
  - Line 1306, Column 25
  - Line 1307, Column 40
  - Line 1310, Column 40
  - Line 1312, Column 37
  - Line 1316, Column 47
  - ... and 10 more occurrences

- **model_a** (appears 3 times):
  - Line 1404, Column 17
  - Line 1408, Column 20
  - Line 1371, Column 70

- **model_b** (appears 3 times):
  - Line 1404, Column 51
  - Line 1409, Column 20
  - Line 1372, Column 70

- **model_id** (appears 8 times):
  - Line 796, Column 24
  - Line 1468, Column 28
  - Line 1521, Column 63
  - Line 795, Column 32
  - Line 1515, Column 80
  - ... and 3 more occurrences

- **model_name** (appears 3 times):
  - Line 1251, Column 36
  - Line 1252, Column 38
  - Line 1253, Column 42

- **name** (appears 2 times):
  - Line 150, Column 53
  - Line 146, Column 38

- **p** (appears 7 times):
  - Line 1371, Column 29
  - Line 1372, Column 29
  - Line 1428, Column 28
  - Line 543, Column 24
  - Line 1371, Column 56
  - ... and 2 more occurrences

- **params** (appears 1 times):
  - Line 646, Column 16

- **proc** (appears 10 times):
  - Line 1875, Column 16
  - Line 1602, Column 68
  - Line 1891, Column 68
  - Line 1602, Column 43
  - Line 1876, Column 31
  - ... and 5 more occurrences

- **request** (appears 25 times):
  - Line 610, Column 22
  - Line 611, Column 25
  - Line 1340, Column 53
  - Line 596, Column 49
  - Line 1015, Column 21
  - ... and 20 more occurrences

- **symbols** (appears 1 times):
  - Line 1140, Column 26

- **t** (appears 10 times):
  - Line 836, Column 30
  - Line 838, Column 24
  - Line 836, Column 51
  - Line 840, Column 16
  - Line 850, Column 33
  - ... and 5 more occurrences

- **token_configs** (appears 10 times):
  - Line 1190, Column 8
  - Line 1217, Column 24
  - Line 1339, Column 24
  - Line 1142, Column 32
  - Line 1218, Column 12
  - ... and 5 more occurrences

- **trade_id** (appears 1 times):
  - Line 871, Column 24

---

### 28. src/analyst/ml_confidence_predictor.py

**Total Issues:** 212

**Issues by Name:**

- **ExecutionRequest** (appears 1 times):
  - Line 1383, Column 32

- **ExecutionStrategy** (appears 7 times):
  - Line 1381, Column 41
  - Line 1381, Column 79
  - Line 1381, Column 112
  - Line 1381, Column 144
  - Line 1381, Column 179
  - ... and 2 more occurrences

- **OrderType** (appears 2 times):
  - Line 1380, Column 25
  - Line 1380, Column 55

- **adverse_level** (appears 3 times):
  - Line 1053, Column 106
  - Line 1090, Column 70
  - Line 1054, Column 45

- **all_conf** (appears 4 times):
  - Line 1642, Column 15
  - Line 1708, Column 15
  - Line 1641, Column 16
  - Line 1707, Column 16

- **analysis** (appears 2 times):
  - Line 1209, Column 25
  - Line 1211, Column 95

- **breakout_price** (appears 1 times):
  - Line 1285, Column 198

- **e** (appears 82 times):
  - Line 294, Column 186
  - Line 514, Column 33
  - Line 1291, Column 51
  - Line 1319, Column 51
  - Line 1388, Column 51
  - ... and 77 more occurrences

- **ensemble_model** (appears 4 times):
  - Line 438, Column 31
  - Line 441, Column 33
  - Line 439, Column 37
  - Line 442, Column 37

- **ensemble_name** (appears 4 times):
  - Line 440, Column 45
  - Line 443, Column 45
  - Line 447, Column 76
  - Line 445, Column 62

- **ensemble_weights** (appears 1 times):
  - Line 893, Column 36

- **f** (appears 1 times):
  - Line 802, Column 44

- **fname** (appears 3 times):
  - Line 799, Column 44
  - Line 803, Column 27
  - Line 797, Column 23

- **force_training** (appears 1 times):
  - Line 1428, Column 19

- **handle_specific_errors** (appears 1 times):
  - Line 89, Column 5

- **inten** (appears 1 times):
  - Line 1607, Column 45

- **k** (appears 10 times):
  - Line 1537, Column 22
  - Line 1611, Column 20
  - Line 1511, Column 41
  - Line 1513, Column 41
  - Line 1527, Column 41
  - ... and 5 more occurrences

- **kwargs** (appears 3 times):
  - Line 1383, Column 277
  - Line 1285, Column 243
  - Line 1313, Column 207

- **level_str** (appears 7 times):
  - Line 1141, Column 27
  - Line 1140, Column 26
  - Line 1164, Column 26
  - Line 262, Column 30
  - Line 271, Column 30
  - ... and 2 more occurrences

- **leverage** (appears 3 times):
  - Line 1385, Column 181
  - Line 1383, Column 157
  - Line 1313, Column 168

- **m** (appears 1 times):
  - Line 1540, Column 29

- **magnitude** (appears 4 times):
  - Line 1211, Column 45
  - Line 1055, Column 237
  - Line 1053, Column 95
  - Line 1055, Column 40

- **member** (appears 2 times):
  - Line 1542, Column 20
  - Line 1537, Column 76

- **model_name** (appears 13 times):
  - Line 684, Column 30
  - Line 1516, Column 19
  - Line 666, Column 43
  - Line 905, Column 41
  - Line 906, Column 41
  - ... and 8 more occurrences

- **name** (appears 1 times):
  - Line 893, Column 57

- **p** (appears 1 times):
  - Line 1467, Column 30

- **param** (appears 1 times):
  - Line 862, Column 19

- **performance_metrics** (appears 1 times):
  - Line 1478, Column 91

- **price** (appears 4 times):
  - Line 1380, Column 44
  - Line 1385, Column 162
  - Line 1383, Column 139
  - Line 1313, Column 150

- **prob** (appears 2 times):
  - Line 265, Column 33
  - Line 274, Column 35

- **quantity** (appears 4 times):
  - Line 1385, Column 143
  - Line 1383, Column 121
  - Line 1285, Column 140
  - Line 1313, Column 132

- **scores** (appears 5 times):
  - Line 1608, Column 12
  - Line 1615, Column 24
  - Line 1609, Column 29
  - Line 1610, Column 28
  - Line 1613, Column 23

- **side** (appears 4 times):
  - Line 1385, Column 125
  - Line 1284, Column 42
  - Line 1312, Column 42
  - Line 1379, Column 42

- **strategy_type** (appears 3 times):
  - Line 1382, Column 50
  - Line 1385, Column 84
  - Line 1383, Column 183

- **symbol** (appears 4 times):
  - Line 1385, Column 109
  - Line 1383, Column 58
  - Line 1285, Column 102
  - Line 1313, Column 94

- **t** (appears 1 times):
  - Line 1610, Column 60

- **target** (appears 3 times):
  - Line 163, Column 94
  - Line 162, Column 40
  - Line 164, Column 40

- **timeframe** (appears 6 times):
  - Line 1557, Column 13
  - Line 1655, Column 13
  - Line 606, Column 31
  - Line 628, Column 109
  - Line 605, Column 107
  - ... and 1 more occurrences

- **training_data** (appears 1 times):
  - Line 1430, Column 158

- **training_type** (appears 3 times):
  - Line 1430, Column 190
  - Line 1436, Column 58
  - Line 1434, Column 101

- **v** (appears 4 times):
  - Line 1537, Column 25
  - Line 1511, Column 44
  - Line 1527, Column 44
  - Line 1626, Column 36

- **val** (appears 2 times):
  - Line 1641, Column 50
  - Line 1707, Column 50

- **x** (appears 1 times):
  - Line 1090, Column 66

---

### 29. src/trading/reporting/trade_analyzer.py

**Total Issues:** 209

**Issues by Name:**

- **e** (appears 12 times):
  - Line 84, Column 33
  - Line 83, Column 74
  - Line 140, Column 69
  - Line 200, Column 71
  - Line 258, Column 64
  - ... and 7 more occurrences

- **f** (appears 6 times):
  - Line 468, Column 38
  - Line 469, Column 38
  - Line 220, Column 51
  - Line 221, Column 51
  - Line 472, Column 59
  - ... and 1 more occurrences

- **i** (appears 2 times):
  - Line 468, Column 41
  - Line 469, Column 41

- **include_explanations** (appears 2 times):
  - Line 72, Column 15
  - Line 561, Column 53

- **info** (appears 1 times):
  - Line 423, Column 30

- **mid** (appears 1 times):
  - Line 190, Column 71

- **model_id** (appears 10 times):
  - Line 149, Column 57
  - Line 150, Column 57
  - Line 151, Column 49
  - Line 172, Column 31
  - Line 460, Column 19
  - ... and 5 more occurrences

- **model_info** (appears 1 times):
  - Line 173, Column 34

- **p** (appears 1 times):
  - Line 190, Column 43

- **score** (appears 3 times):
  - Line 528, Column 15
  - Line 533, Column 17
  - Line 538, Column 17

- **trade** (appears 154 times):
  - Line 561, Column 46
  - Line 89, Column 24
  - Line 91, Column 22
  - Line 92, Column 22
  - Line 93, Column 24
  - ... and 149 more occurrences

- **v** (appears 8 times):
  - Line 220, Column 54
  - Line 221, Column 54
  - Line 223, Column 52
  - Line 235, Column 52
  - Line 220, Column 88
  - ... and 3 more occurrences

- **w** (appears 3 times):
  - Line 433, Column 34
  - Line 433, Column 75
  - Line 433, Column 45

- **x** (appears 5 times):
  - Line 241, Column 88
  - Line 217, Column 84
  - Line 230, Column 84
  - Line 468, Column 116
  - Line 469, Column 116

---

### 30. src/utils/ml_common/optimization/hpo_utils.py

**Total Issues:** 208

**Issues by Name:**

- **cfg** (appears 7 times):
  - Line 2190, Column 37
  - Line 2193, Column 30
  - Line 2204, Column 37
  - Line 2202, Column 41
  - Line 2194, Column 26
  - ... and 2 more occurrences

- **combination** (appears 1 times):
  - Line 2576, Column 30

- **combo** (appears 2 times):
  - Line 2163, Column 54
  - Line 2508, Column 54

- **e** (appears 67 times):
  - Line 52, Column 50
  - Line 94, Column 59
  - Line 2742, Column 29
  - Line 234, Column 33
  - Line 444, Column 33
  - ... and 62 more occurrences

- **executor** (appears 1 times):
  - Line 1338, Column 35

- **fold_scores** (appears 8 times):
  - Line 2097, Column 19
  - Line 921, Column 23
  - Line 2096, Column 20
  - Line 2098, Column 41
  - Line 916, Column 24
  - ... and 3 more occurrences

- **i** (appears 7 times):
  - Line 917, Column 56
  - Line 2363, Column 24
  - Line 2414, Column 24
  - Line 2606, Column 24
  - Line 2364, Column 58
  - ... and 2 more occurrences

- **k** (appears 7 times):
  - Line 2178, Column 21
  - Line 2178, Column 43
  - Line 2178, Column 73
  - Line 2225, Column 33
  - Line 2227, Column 33
  - ... and 2 more occurrences

- **kwargs** (appears 12 times):
  - Line 2901, Column 18
  - Line 2896, Column 29
  - Line 2897, Column 38
  - Line 2712, Column 18
  - Line 249, Column 33
  - ... and 7 more occurrences

- **name** (appears 12 times):
  - Line 2158, Column 29
  - Line 2189, Column 19
  - Line 2192, Column 36
  - Line 2464, Column 19
  - Line 2467, Column 39
  - ... and 7 more occurrences

- **obj** (appears 5 times):
  - Line 632, Column 23
  - Line 634, Column 27
  - Line 636, Column 29
  - Line 635, Column 60
  - Line 637, Column 59

- **param** (appears 6 times):
  - Line 1243, Column 20
  - Line 1247, Column 34
  - Line 1842, Column 23
  - Line 1843, Column 52
  - Line 1858, Column 40
  - ... and 1 more occurrences

- **param_name** (appears 28 times):
  - Line 2525, Column 15
  - Line 2528, Column 37
  - Line 1899, Column 31
  - Line 1900, Column 35
  - Line 1913, Column 31
  - ... and 23 more occurrences

- **r** (appears 2 times):
  - Line 1345, Column 77
  - Line 1346, Column 69

- **space** (appears 1 times):
  - Line 204, Column 71

- **t** (appears 17 times):
  - Line 456, Column 33
  - Line 457, Column 29
  - Line 2247, Column 25
  - Line 304, Column 32
  - Line 464, Column 36
  - ... and 12 more occurrences

- **task** (appears 1 times):
  - Line 1340, Column 51

- **task_e** (appears 1 times):
  - Line 1333, Column 41

- **test_idx** (appears 4 times):
  - Line 2081, Column 49
  - Line 2082, Column 49
  - Line 893, Column 53
  - Line 894, Column 53

- **train_idx** (appears 6 times):
  - Line 2081, Column 35
  - Line 2082, Column 35
  - Line 893, Column 39
  - Line 894, Column 39
  - Line 2086, Column 90
  - ... and 1 more occurrences

- **v** (appears 13 times):
  - Line 2222, Column 34
  - Line 323, Column 23
  - Line 2224, Column 38
  - Line 2223, Column 46
  - Line 2223, Column 32
  - ... and 8 more occurrences

---

### 31. src/feature_generation/categories/volume.py

**Total Issues:** 200

**Issues by Name:**

- **args** (appears 2 times):
  - Line 68, Column 15
  - Line 630, Column 37

- **col** (appears 4 times):
  - Line 2486, Column 33
  - Line 2584, Column 33
  - Line 2693, Column 33
  - Line 2780, Column 33

- **df** (appears 6 times):
  - Line 934, Column 74
  - Line 944, Column 84
  - Line 840, Column 53
  - Line 851, Column 53
  - Line 829, Column 53
  - ... and 1 more occurrences

- **e** (appears 78 times):
  - Line 306, Column 72
  - Line 423, Column 71
  - Line 503, Column 58
  - Line 535, Column 55
  - Line 562, Column 61
  - ... and 73 more occurrences

- **i** (appears 61 times):
  - Line 975, Column 17
  - Line 3084, Column 15
  - Line 3249, Column 15
  - Line 3040, Column 30
  - Line 3081, Column 33
  - ... and 56 more occurrences

- **j** (appears 15 times):
  - Line 3136, Column 24
  - Line 3136, Column 34
  - Line 3252, Column 23
  - Line 3140, Column 45
  - Line 3137, Column 86
  - ... and 10 more occurrences

- **kwargs** (appears 23 times):
  - Line 3349, Column 52
  - Line 3472, Column 50
  - Line 3625, Column 80
  - Line 68, Column 23
  - Line 630, Column 45
  - ... and 18 more occurrences

- **op** (appears 3 times):
  - Line 741, Column 29
  - Line 741, Column 48
  - Line 741, Column 67

- **price_col** (appears 3 times):
  - Line 889, Column 19
  - Line 897, Column 44
  - Line 891, Column 42

- **w** (appears 5 times):
  - Line 851, Column 77
  - Line 934, Column 78
  - Line 944, Column 88
  - Line 829, Column 75
  - Line 840, Column 89

---

### 32. src/feature_generation/utils/feature_generators.py

**Total Issues:** 198

**Issues by Name:**

- **c** (appears 12 times):
  - Line 381, Column 23
  - Line 393, Column 24
  - Line 462, Column 23
  - Line 567, Column 30
  - Line 507, Column 27
  - ... and 7 more occurrences

- **col** (appears 28 times):
  - Line 2216, Column 27
  - Line 2217, Column 25
  - Line 2225, Column 20
  - Line 2226, Column 21
  - Line 2050, Column 33
  - ... and 23 more occurrences

- **e** (appears 50 times):
  - Line 31, Column 65
  - Line 51, Column 58
  - Line 60, Column 44
  - Line 109, Column 86
  - Line 621, Column 72
  - ... and 45 more occurrences

- **e2** (appears 1 times):
  - Line 296, Column 100

- **feature_values** (appears 1 times):
  - Line 220, Column 52

- **group_configs** (appears 8 times):
  - Line 291, Column 30
  - Line 271, Column 102
  - Line 273, Column 98
  - Line 275, Column 97
  - Line 277, Column 95
  - ... and 3 more occurrences

- **group_name** (appears 7 times):
  - Line 270, Column 19
  - Line 272, Column 21
  - Line 274, Column 21
  - Line 276, Column 21
  - Line 289, Column 63
  - ... and 2 more occurrences

- **i** (appears 6 times):
  - Line 842, Column 29
  - Line 837, Column 58
  - Line 838, Column 55
  - Line 837, Column 43
  - Line 838, Column 40
  - ... and 1 more occurrences

- **indicator_name** (appears 15 times):
  - Line 191, Column 19
  - Line 188, Column 48
  - Line 193, Column 21
  - Line 195, Column 21
  - Line 223, Column 51
  - ... and 10 more occurrences

- **k** (appears 1 times):
  - Line 336, Column 16

- **key** (appears 2 times):
  - Line 2147, Column 30
  - Line 2153, Column 30

- **keyword** (appears 6 times):
  - Line 319, Column 19
  - Line 321, Column 21
  - Line 323, Column 21
  - Line 325, Column 21
  - Line 327, Column 21
  - ... and 1 more occurrences

- **kwargs** (appears 38 times):
  - Line 2329, Column 10
  - Line 2337, Column 52
  - Line 2341, Column 52
  - Line 2345, Column 52
  - Line 2349, Column 64
  - ... and 33 more occurrences

- **macd_col** (appears 3 times):
  - Line 2230, Column 51
  - Line 2231, Column 97
  - Line 2231, Column 42

- **mom_col** (appears 3 times):
  - Line 2221, Column 51
  - Line 2222, Column 96
  - Line 2222, Column 42

- **rsi_col** (appears 3 times):
  - Line 2230, Column 19
  - Line 2231, Column 77
  - Line 2231, Column 32

- **start_idx** (appears 4 times):
  - Line 642, Column 33
  - Line 645, Column 31
  - Line 640, Column 34
  - Line 643, Column 53

- **v** (appears 2 times):
  - Line 336, Column 19
  - Line 336, Column 51

- **values** (appears 2 times):
  - Line 2147, Column 37
  - Line 2153, Column 37

- **vol_col** (appears 3 times):
  - Line 2221, Column 19
  - Line 2222, Column 76
  - Line 2222, Column 32

- **warnings** (appears 1 times):
  - Line 2488, Column 4

- **x** (appears 2 times):
  - Line 1773, Column 91
  - Line 1774, Column 89

---

### 33. src/training/steps/pre_training/final_feature_selection_step.py

**Total Issues:** 195

**Issues by Name:**

- **base_name** (appears 6 times):
  - Line 512, Column 24
  - Line 519, Column 45
  - Line 509, Column 90
  - Line 520, Column 68
  - Line 524, Column 75
  - ... and 1 more occurrences

- **col** (appears 59 times):
  - Line 673, Column 47
  - Line 729, Column 37
  - Line 1011, Column 36
  - Line 1096, Column 36
  - Line 781, Column 37
  - ... and 54 more occurrences

- **col_error** (appears 2 times):
  - Line 1676, Column 76
  - Line 1713, Column 76

- **contract_error** (appears 1 times):
  - Line 1475, Column 72

- **count** (appears 2 times):
  - Line 454, Column 31
  - Line 455, Column 62

- **e** (appears 48 times):
  - Line 161, Column 64
  - Line 339, Column 57
  - Line 349, Column 70
  - Line 359, Column 69
  - Line 464, Column 64
  - ... and 43 more occurrences

- **exc** (appears 3 times):
  - Line 212, Column 42
  - Line 213, Column 79
  - Line 608, Column 77

- **expected_exchange** (appears 2 times):
  - Line 619, Column 60
  - Line 626, Column 69

- **expected_symbol** (appears 2 times):
  - Line 618, Column 56
  - Line 623, Column 67

- **expected_timeframe** (appears 2 times):
  - Line 620, Column 62
  - Line 629, Column 70

- **f** (appears 3 times):
  - Line 1894, Column 15
  - Line 1924, Column 36
  - Line 556, Column 79

- **fallback_error** (appears 1 times):
  - Line 789, Column 61

- **feature** (appears 5 times):
  - Line 2130, Column 53
  - Line 2131, Column 42
  - Line 2115, Column 36
  - Line 2116, Column 32
  - Line 2045, Column 41

- **filename** (appears 7 times):
  - Line 832, Column 40
  - Line 955, Column 40
  - Line 833, Column 83
  - Line 956, Column 97
  - Line 854, Column 61
  - ... and 2 more occurrences

- **handle** (appears 1 times):
  - Line 601, Column 41

- **i** (appears 4 times):
  - Line 509, Column 56
  - Line 833, Column 54
  - Line 956, Column 61
  - Line 2045, Column 33

- **importance** (appears 1 times):
  - Line 2045, Column 52

- **issue** (appears 1 times):
  - Line 692, Column 38

- **item** (appears 1 times):
  - Line 2200, Column 33

- **key** (appears 11 times):
  - Line 1833, Column 12
  - Line 1836, Column 12
  - Line 1839, Column 12
  - Line 1404, Column 16
  - Line 1420, Column 16
  - ... and 6 more occurrences

- **kwargs** (appears 1 times):
  - Line 217, Column 38

- **label** (appears 2 times):
  - Line 2031, Column 52
  - Line 2033, Column 52

- **metric_key** (appears 2 times):
  - Line 2028, Column 23
  - Line 2029, Column 69

- **nested_key** (appears 1 times):
  - Line 2184, Column 41

- **nested_value** (appears 9 times):
  - Line 2196, Column 29
  - Line 2190, Column 62
  - Line 2187, Column 38
  - Line 2190, Column 38
  - Line 2192, Column 38
  - ... and 4 more occurrences

- **runtime_config** (appears 5 times):
  - Line 2237, Column 37
  - Line 2217, Column 51
  - Line 2226, Column 14
  - Line 2228, Column 23
  - Line 2222, Column 67

- **strategy** (appears 1 times):
  - Line 455, Column 50

- **target** (appears 9 times):
  - Line 742, Column 19
  - Line 758, Column 44
  - Line 759, Column 44
  - Line 744, Column 38
  - Line 744, Column 67
  - ... and 4 more occurrences

- **vectorbt_error** (appears 1 times):
  - Line 1753, Column 73

- **x** (appears 2 times):
  - Line 763, Column 51
  - Line 2040, Column 34

---

### 34. src/feature_generation/categories/trend.py

**Total Issues:** 192

**Issues by Name:**

- **acc** (appears 1 times):
  - Line 2296, Column 64

- **base_kwargs** (appears 18 times):
  - Line 1167, Column 74
  - Line 1236, Column 74
  - Line 1329, Column 74
  - Line 1402, Column 74
  - Line 1472, Column 74
  - ... and 13 more occurrences

- **col** (appears 5 times):
  - Line 690, Column 25
  - Line 690, Column 51
  - Line 2450, Column 33
  - Line 2555, Column 33
  - Line 2668, Column 33

- **df** (appears 4 times):
  - Line 757, Column 73
  - Line 769, Column 24
  - Line 769, Column 43
  - Line 769, Column 61

- **e** (appears 48 times):
  - Line 806, Column 62
  - Line 1971, Column 66
  - Line 2001, Column 63
  - Line 2515, Column 66
  - Line 2628, Column 65
  - ... and 43 more occurrences

- **e2** (appears 2 times):
  - Line 281, Column 75
  - Line 621, Column 105

- **feature** (appears 1 times):
  - Line 1995, Column 41

- **i** (appears 46 times):
  - Line 2686, Column 41
  - Line 2687, Column 39
  - Line 2578, Column 31
  - Line 2579, Column 35
  - Line 2580, Column 29
  - ... and 41 more occurrences

- **kijun** (appears 1 times):
  - Line 2291, Column 73

- **kwargs** (appears 47 times):
  - Line 2817, Column 52
  - Line 788, Column 77
  - Line 1941, Column 63
  - Line 1943, Column 66
  - Line 1960, Column 18
  - ... and 42 more occurrences

- **macd_config** (appears 3 times):
  - Line 3239, Column 26
  - Line 3240, Column 26
  - Line 3241, Column 28

- **max_acc** (appears 1 times):
  - Line 2296, Column 69

- **p** (appears 1 times):
  - Line 769, Column 81

- **tenkan** (appears 1 times):
  - Line 2291, Column 65

- **w** (appears 2 times):
  - Line 1995, Column 37
  - Line 757, Column 93

- **window_groups** (appears 1 times):
  - Line 563, Column 38

- **x** (appears 10 times):
  - Line 1379, Column 37
  - Line 1366, Column 41
  - Line 1373, Column 41
  - Line 454, Column 61
  - Line 454, Column 77
  - ... and 5 more occurrences

---

### 35. src/training/steps/market_analysis/shared_utils/balanced_feature_extractor.py

**Total Issues:** 187

**Issues by Name:**

- **arr** (appears 3 times):
  - Line 460, Column 47
  - Line 451, Column 33
  - Line 459, Column 38

- **category** (appears 16 times):
  - Line 249, Column 34
  - Line 260, Column 23
  - Line 262, Column 25
  - Line 257, Column 42
  - Line 264, Column 25
  - ... and 11 more occurrences

- **e** (appears 48 times):
  - Line 409, Column 26
  - Line 435, Column 26
  - Line 374, Column 63
  - Line 470, Column 74
  - Line 491, Column 61
  - ... and 43 more occurrences

- **feature_name** (appears 2 times):
  - Line 1551, Column 39
  - Line 1552, Column 57

- **i** (appears 29 times):
  - Line 1487, Column 34
  - Line 486, Column 27
  - Line 506, Column 27
  - Line 1137, Column 92
  - Line 484, Column 50
  - ... and 24 more occurrences

- **kwargs** (appears 11 times):
  - Line 1650, Column 77
  - Line 1693, Column 61
  - Line 1696, Column 62
  - Line 1654, Column 59
  - Line 1669, Column 77
  - ... and 6 more occurrences

- **label** (appears 2 times):
  - Line 1462, Column 33
  - Line 1514, Column 33

- **name** (appears 5 times):
  - Line 1137, Column 30
  - Line 1422, Column 44
  - Line 1423, Column 45
  - Line 1539, Column 51
  - Line 1551, Column 31

- **period** (appears 36 times):
  - Line 703, Column 39
  - Line 879, Column 39
  - Line 953, Column 39
  - Line 786, Column 42
  - Line 834, Column 42
  - ... and 31 more occurrences

- **profile** (appears 1 times):
  - Line 1544, Column 27

- **q** (appears 2 times):
  - Line 631, Column 61
  - Line 635, Column 53

- **regime_id** (appears 2 times):
  - Line 1401, Column 47
  - Line 1411, Column 50

- **regime_name** (appears 2 times):
  - Line 1556, Column 55
  - Line 1558, Column 55

- **rolling_apply** (appears 1 times):
  - Line 1696, Column 19

- **rolling_max** (appears 1 times):
  - Line 1662, Column 23

- **rolling_min** (appears 1 times):
  - Line 1660, Column 23

- **rolling_sum** (appears 1 times):
  - Line 1664, Column 23

- **rolling_var** (appears 1 times):
  - Line 1658, Column 23

- **score** (appears 2 times):
  - Line 1539, Column 58
  - Line 1552, Column 72

- **short_period** (appears 12 times):
  - Line 1220, Column 42
  - Line 1250, Column 42
  - Line 1225, Column 39
  - Line 1254, Column 33
  - Line 1237, Column 57
  - ... and 7 more occurrences

- **x** (appears 9 times):
  - Line 1397, Column 50
  - Line 541, Column 42
  - Line 541, Column 26
  - Line 570, Column 46
  - Line 570, Column 26
  - ... and 4 more occurrences

---

### 36. src/training/steps/market_analysis/clusters/nas_tas_clustering_refactored.py

**Total Issues:** 179

**Issues by Name:**

- **array** (appears 6 times):
  - Line 1632, Column 27
  - Line 1642, Column 27
  - Line 1632, Column 47
  - Line 1633, Column 20
  - Line 1642, Column 47
  - ... and 1 more occurrences

- **cluster** (appears 9 times):
  - Line 1428, Column 36
  - Line 1412, Column 46
  - Line 1415, Column 40
  - Line 1423, Column 40
  - Line 1585, Column 51
  - ... and 4 more occurrences

- **cluster1** (appears 1 times):
  - Line 2097, Column 59

- **cluster2** (appears 1 times):
  - Line 2097, Column 47

- **e** (appears 97 times):
  - Line 480, Column 37
  - Line 1748, Column 33
  - Line 2154, Column 29
  - Line 294, Column 41
  - Line 295, Column 51
  - ... and 92 more occurrences

- **i** (appears 31 times):
  - Line 816, Column 40
  - Line 883, Column 43
  - Line 915, Column 41
  - Line 945, Column 41
  - Line 1060, Column 56
  - ... and 26 more occurrences

- **item** (appears 7 times):
  - Line 1149, Column 30
  - Line 1150, Column 37
  - Line 1152, Column 73
  - Line 1152, Column 53
  - Line 1101, Column 42
  - ... and 2 more occurrences

- **j** (appears 2 times):
  - Line 2090, Column 48
  - Line 2091, Column 80

- **key** (appears 5 times):
  - Line 1116, Column 30
  - Line 1118, Column 30
  - Line 1132, Column 31
  - Line 1135, Column 35
  - Line 1137, Column 35

- **keyword** (appears 4 times):
  - Line 1759, Column 19
  - Line 1761, Column 21
  - Line 1763, Column 21
  - Line 1765, Column 21

- **regime** (appears 3 times):
  - Line 1451, Column 50
  - Line 1667, Column 58
  - Line 1689, Column 66

- **stats** (appears 2 times):
  - Line 1429, Column 88
  - Line 1429, Column 52

- **umap** (appears 1 times):
  - Line 1968, Column 22

- **value** (appears 9 times):
  - Line 1116, Column 43
  - Line 1131, Column 34
  - Line 1097, Column 34
  - Line 1097, Column 64
  - Line 1098, Column 30
  - ... and 4 more occurrences

- **warning** (appears 1 times):
  - Line 280, Column 45

---

### 37. src/utils/ml_common/models/model_factory.py

**Total Issues:** 179

**Issues by Name:**

- **data** (appears 8 times):
  - Line 1487, Column 38
  - Line 1468, Column 36
  - Line 1473, Column 35
  - Line 1475, Column 40
  - Line 1471, Column 40
  - ... and 3 more occurrences

- **e** (appears 14 times):
  - Line 2644, Column 97
  - Line 2681, Column 108
  - Line 428, Column 114
  - Line 2528, Column 83
  - Line 2567, Column 83
  - ... and 9 more occurrences

- **key** (appears 3 times):
  - Line 1774, Column 37
  - Line 1776, Column 32
  - Line 1775, Column 38

- **kwargs** (appears 127 times):
  - Line 777, Column 30
  - Line 830, Column 30
  - Line 901, Column 30
  - Line 1047, Column 30
  - Line 1567, Column 30
  - ... and 122 more occurrences

- **r** (appears 6 times):
  - Line 1507, Column 36
  - Line 1510, Column 36
  - Line 1508, Column 58
  - Line 1508, Column 79
  - Line 1511, Column 51
  - ... and 1 more occurrences

- **regime** (appears 17 times):
  - Line 1266, Column 23
  - Line 1472, Column 44
  - Line 1364, Column 49
  - Line 1377, Column 44
  - Line 1377, Column 87
  - ... and 12 more occurrences

- **value** (appears 2 times):
  - Line 1776, Column 39
  - Line 1775, Column 43

- **x_sample** (appears 2 times):
  - Line 1436, Column 71
  - Line 1440, Column 63

---

### 38. src/trading/reporting/dashboard_generator.py

**Total Issues:** 177

**Issues by Name:**

- **dashboard_name** (appears 4 times):
  - Line 477, Column 59
  - Line 467, Column 54
  - Line 472, Column 57
  - Line 610, Column 54

- **data** (appears 7 times):
  - Line 371, Column 19
  - Line 374, Column 20
  - Line 372, Column 20
  - Line 373, Column 20
  - Line 372, Column 53
  - ... and 2 more occurrences

- **e** (appears 10 times):
  - Line 105, Column 67
  - Line 175, Column 65
  - Line 231, Column 71
  - Line 284, Column 68
  - Line 338, Column 67
  - ... and 5 more occurrences

- **f** (appears 3 times):
  - Line 469, Column 37
  - Line 474, Column 37
  - Line 612, Column 16

- **metrics** (appears 3 times):
  - Line 777, Column 30
  - Line 778, Column 29
  - Line 779, Column 29

- **model_id** (appears 6 times):
  - Line 261, Column 36
  - Line 785, Column 25
  - Line 246, Column 53
  - Line 258, Column 59
  - Line 259, Column 66
  - ... and 1 more occurrences

- **p** (appears 2 times):
  - Line 271, Column 45
  - Line 271, Column 69

- **session_metrics** (appears 8 times):
  - Line 811, Column 69
  - Line 84, Column 74
  - Line 162, Column 64
  - Line 163, Column 116
  - Line 164, Column 64
  - ... and 3 more occurrences

- **t** (appears 31 times):
  - Line 119, Column 29
  - Line 291, Column 29
  - Line 345, Column 29
  - Line 246, Column 32
  - Line 126, Column 38
  - ... and 26 more occurrences

- **trade** (appears 66 times):
  - Line 214, Column 25
  - Line 219, Column 19
  - Line 349, Column 25
  - Line 355, Column 25
  - Line 365, Column 19
  - ... and 61 more occurrences

- **trade_id** (appears 1 times):
  - Line 405, Column 32

- **trades** (appears 33 times):
  - Line 122, Column 15
  - Line 141, Column 15
  - Line 188, Column 25
  - Line 200, Column 25
  - Line 213, Column 25
  - ... and 28 more occurrences

- **x** (appears 3 times):
  - Line 421, Column 61
  - Line 383, Column 88
  - Line 384, Column 84

---

### 39. src/training/steps/market_analysis/hybrid_nas_tas_regime/hybrid_orchestrator.py

**Total Issues:** 176

**Issues by Name:**

- **args** (appears 1 times):
  - Line 25, Column 19

- **assignment** (appears 2 times):
  - Line 2531, Column 30
  - Line 2531, Column 62

- **attr** (appears 8 times):
  - Line 2155, Column 61
  - Line 2158, Column 27
  - Line 2160, Column 52
  - Line 2161, Column 64
  - Line 2162, Column 60
  - ... and 3 more occurrences

- **col** (appears 4 times):
  - Line 2355, Column 49
  - Line 1443, Column 53
  - Line 2355, Column 81
  - Line 1443, Column 85

- **count** (appears 9 times):
  - Line 1831, Column 48
  - Line 3563, Column 69
  - Line 3573, Column 42
  - Line 1830, Column 30
  - Line 2537, Column 30
  - ... and 4 more occurrences

- **d** (appears 3 times):
  - Line 3183, Column 31
  - Line 3182, Column 42
  - Line 3182, Column 46

- **duration** (appears 3 times):
  - Line 3285, Column 30
  - Line 3290, Column 30
  - Line 3285, Column 60

- **e** (appears 95 times):
  - Line 634, Column 29
  - Line 745, Column 29
  - Line 841, Column 29
  - Line 1043, Column 63
  - Line 1591, Column 86
  - ... and 90 more occurrences

- **from_regime** (appears 2 times):
  - Line 3561, Column 19
  - Line 3562, Column 56

- **i** (appears 19 times):
  - Line 924, Column 45
  - Line 925, Column 45
  - Line 2616, Column 19
  - Line 2617, Column 42
  - Line 3168, Column 31
  - ... and 14 more occurrences

- **j** (appears 4 times):
  - Line 2616, Column 41
  - Line 2618, Column 42
  - Line 2605, Column 41
  - Line 2619, Column 54

- **kwargs** (appears 1 times):
  - Line 25, Column 27

- **metrics** (appears 4 times):
  - Line 1947, Column 38
  - Line 1948, Column 53
  - Line 1948, Column 83
  - Line 1948, Column 119

- **regime** (appears 12 times):
  - Line 1780, Column 47
  - Line 2391, Column 55
  - Line 1831, Column 33
  - Line 2272, Column 40
  - Line 2287, Column 40
  - ... and 7 more occurrences

- **regime_key** (appears 1 times):
  - Line 1948, Column 34

- **run_hybrid_orchestrator_example** (appears 1 times):
  - Line 3956, Column 16

- **size** (appears 3 times):
  - Line 3090, Column 35
  - Line 3095, Column 35
  - Line 3088, Column 47

- **to_regime** (appears 2 times):
  - Line 3561, Column 53
  - Line 3562, Column 85

- **x** (appears 2 times):
  - Line 2820, Column 37
  - Line 3081, Column 67

---

### 40. src/utils/data/quality/data_quality.py

**Total Issues:** 176

**Issues by Name:**

- **c** (appears 2 times):
  - Line 776, Column 39
  - Line 776, Column 79

- **calc** (appears 4 times):
  - Line 303, Column 37
  - Line 311, Column 63
  - Line 370, Column 74
  - Line 389, Column 79

- **col** (appears 82 times):
  - Line 311, Column 26
  - Line 556, Column 25
  - Line 752, Column 28
  - Line 297, Column 31
  - Line 299, Column 22
  - ... and 77 more occurrences

- **column** (appears 14 times):
  - Line 1019, Column 19
  - Line 1031, Column 19
  - Line 1055, Column 19
  - Line 1032, Column 39
  - Line 1020, Column 43
  - ... and 9 more occurrences

- **constraints** (appears 10 times):
  - Line 1033, Column 32
  - Line 1042, Column 32
  - Line 1033, Column 48
  - Line 1042, Column 48
  - Line 1034, Column 56
  - ... and 5 more occurrences

- **e** (appears 9 times):
  - Line 670, Column 75
  - Line 832, Column 101
  - Line 890, Column 69
  - Line 891, Column 100
  - Line 930, Column 70
  - ... and 4 more occurrences

- **expected_type** (appears 3 times):
  - Line 1021, Column 38
  - Line 1026, Column 93
  - Line 1025, Column 93

- **feature** (appears 8 times):
  - Line 528, Column 33
  - Line 529, Column 36
  - Line 530, Column 47
  - Line 547, Column 33
  - Line 548, Column 36
  - ... and 3 more occurrences

- **fields** (appears 4 times):
  - Line 1317, Column 35
  - Line 1312, Column 44
  - Line 1319, Column 43
  - Line 1318, Column 60

- **group** (appears 1 times):
  - Line 749, Column 37

- **i** (appears 7 times):
  - Line 562, Column 26
  - Line 793, Column 31
  - Line 565, Column 45
  - Line 569, Column 45
  - Line 571, Column 45
  - ... and 2 more occurrences

- **issue** (appears 14 times):
  - Line 1375, Column 29
  - Line 677, Column 29
  - Line 940, Column 43
  - Line 985, Column 31
  - Line 1375, Column 80
  - ... and 9 more occurrences

- **j** (appears 2 times):
  - Line 794, Column 53
  - Line 798, Column 56

- **ops** (appears 1 times):
  - Line 1318, Column 44

- **other** (appears 4 times):
  - Line 762, Column 51
  - Line 765, Column 55
  - Line 768, Column 67
  - Line 771, Column 53

- **pair** (appears 6 times):
  - Line 816, Column 42
  - Line 810, Column 33
  - Line 810, Column 47
  - Line 822, Column 43
  - Line 822, Column 60
  - ... and 1 more occurrences

- **recommendation** (appears 1 times):
  - Line 885, Column 64

- **warning** (appears 3 times):
  - Line 1376, Column 31
  - Line 946, Column 40
  - Line 1376, Column 88

- **x** (appears 1 times):
  - Line 1319, Column 77

---

### 41. src/trading/monitoring/comprehensive_trade_monitor.py

**Total Issues:** 175

**Issues by Name:**

- **EnhancedMonitoringConfig** (appears 1 times):
  - Line 213, Column 32

- **EnhancedMonitoringOrchestrator** (appears 1 times):
  - Line 180, Column 35

- **ExplainabilityIntegrator** (appears 1 times):
  - Line 181, Column 41

- **ExplainabilityOrchestrator** (appears 1 times):
  - Line 182, Column 43

- **col** (appears 3 times):
  - Line 595, Column 19
  - Line 596, Column 29
  - Line 596, Column 54

- **e** (appears 23 times):
  - Line 249, Column 82
  - Line 365, Column 65
  - Line 402, Column 74
  - Line 419, Column 75
  - Line 436, Column 84
  - ... and 18 more occurrences

- **export_format** (appears 4 times):
  - Line 1145, Column 86
  - Line 1036, Column 15
  - Line 1045, Column 17
  - Line 913, Column 75

- **f** (appears 2 times):
  - Line 740, Column 51
  - Line 1041, Column 38

- **feature** (appears 3 times):
  - Line 641, Column 23
  - Line 648, Column 53
  - Line 643, Column 53

- **k** (appears 1 times):
  - Line 618, Column 24

- **key** (appears 2 times):
  - Line 1066, Column 59
  - Line 1066, Column 38

- **kwargs** (appears 11 times):
  - Line 1157, Column 77
  - Line 1200, Column 61
  - Line 1203, Column 62
  - Line 1161, Column 59
  - Line 1176, Column 77
  - ... and 6 more occurrences

- **model_id** (appears 17 times):
  - Line 378, Column 42
  - Line 388, Column 52
  - Line 391, Column 52
  - Line 394, Column 48
  - Line 397, Column 49
  - ... and 12 more occurrences

- **model_info** (appears 18 times):
  - Line 387, Column 35
  - Line 390, Column 35
  - Line 393, Column 31
  - Line 396, Column 32
  - Line 388, Column 64
  - ... and 13 more occurrences

- **models_used** (appears 8 times):
  - Line 314, Column 15
  - Line 1134, Column 79
  - Line 342, Column 44
  - Line 376, Column 40
  - Line 547, Column 40
  - ... and 3 more occurrences

- **outcome_data** (appears 11 times):
  - Line 1138, Column 76
  - Line 784, Column 39
  - Line 785, Column 41
  - Line 786, Column 43
  - Line 787, Column 45
  - ... and 6 more occurrences

- **position_sizing** (appears 4 times):
  - Line 428, Column 42
  - Line 429, Column 37
  - Line 430, Column 43
  - Line 431, Column 43

- **regime_data** (appears 4 times):
  - Line 411, Column 40
  - Line 412, Column 46
  - Line 413, Column 49
  - Line 414, Column 45

- **report_type** (appears 11 times):
  - Line 1145, Column 73
  - Line 903, Column 15
  - Line 905, Column 17
  - Line 901, Column 43
  - Line 913, Column 62
  - ... and 6 more occurrences

- **risk_metrics** (appears 5 times):
  - Line 445, Column 43
  - Line 446, Column 35
  - Line 447, Column 47
  - Line 448, Column 46
  - Line 449, Column 48

- **shap_values** (appears 3 times):
  - Line 641, Column 34
  - Line 633, Column 36
  - Line 643, Column 41

- **t** (appears 14 times):
  - Line 998, Column 16
  - Line 1007, Column 34
  - Line 1008, Column 33
  - Line 932, Column 30
  - Line 1006, Column 28
  - ... and 9 more occurrences

- **trade_data** (appears 19 times):
  - Line 1134, Column 67
  - Line 318, Column 35
  - Line 320, Column 37
  - Line 322, Column 36
  - Line 326, Column 32
  - ... and 14 more occurrences

- **v** (appears 2 times):
  - Line 618, Column 27
  - Line 618, Column 73

- **value** (appears 5 times):
  - Line 1067, Column 30
  - Line 1068, Column 33
  - Line 1069, Column 32
  - Line 1072, Column 41
  - Line 1070, Column 45

- **warnings** (appears 1 times):
  - Line 709, Column 4

---

### 42. src/training/steps/data_collection/data_consolidation_manager.py

**Total Issues:** 172

**Issues by Name:**

- **batch_file** (appears 5 times):
  - Line 302, Column 72
  - Line 371, Column 24
  - Line 372, Column 49
  - Line 315, Column 65
  - Line 374, Column 71

- **chunk_idx** (appears 3 times):
  - Line 429, Column 44
  - Line 426, Column 58
  - Line 441, Column 56

- **chunk_size** (appears 5 times):
  - Line 258, Column 34
  - Line 261, Column 53
  - Line 415, Column 86
  - Line 418, Column 93
  - Line 418, Column 45

- **consolidate_all_data** (appears 1 times):
  - Line 702, Column 27

- **consolidate_session_data** (appears 1 times):
  - Line 677, Column 23

- **consolidate_time_range_data** (appears 1 times):
  - Line 689, Column 28

- **consolidated_file** (appears 2 times):
  - Line 465, Column 33
  - Line 467, Column 93

- **data_type** (appears 23 times):
  - Line 641, Column 66
  - Line 655, Column 69
  - Line 667, Column 69
  - Line 507, Column 85
  - Line 534, Column 85
  - ... and 18 more occurrences

- **e** (appears 17 times):
  - Line 132, Column 29
  - Line 208, Column 29
  - Line 275, Column 29
  - Line 397, Column 29
  - Line 492, Column 29
  - ... and 12 more occurrences

- **end_time** (appears 6 times):
  - Line 655, Column 103
  - Line 170, Column 68
  - Line 183, Column 148
  - Line 553, Column 23
  - Line 553, Column 49
  - ... and 1 more occurrences

- **exchange** (appears 23 times):
  - Line 641, Column 56
  - Line 655, Column 59
  - Line 667, Column 59
  - Line 92, Column 69
  - Line 97, Column 24
  - ... and 18 more occurrences

- **f** (appears 1 times):
  - Line 457, Column 26

- **file_chunk** (appears 2 times):
  - Line 431, Column 20
  - Line 426, Column 98

- **file_path** (appears 3 times):
  - Line 556, Column 42
  - Line 548, Column 56
  - Line 559, Column 72

- **i** (appears 2 times):
  - Line 418, Column 39
  - Line 418, Column 41

- **kwargs** (appears 3 times):
  - Line 641, Column 102
  - Line 655, Column 115
  - Line 667, Column 93

- **max_memory_mb** (appears 7 times):
  - Line 112, Column 46
  - Line 188, Column 51
  - Line 261, Column 38
  - Line 268, Column 54
  - Line 432, Column 56
  - ... and 2 more occurrences

- **remove_originals** (appears 6 times):
  - Line 368, Column 15
  - Line 112, Column 28
  - Line 188, Column 33
  - Line 261, Column 20
  - Line 268, Column 36
  - ... and 1 more occurrences

- **session_id** (appears 9 times):
  - Line 641, Column 88
  - Line 92, Column 55
  - Line 97, Column 56
  - Line 112, Column 16
  - Line 351, Column 83
  - ... and 4 more occurrences

- **start_time** (appears 6 times):
  - Line 655, Column 91
  - Line 170, Column 56
  - Line 183, Column 84
  - Line 551, Column 23
  - Line 551, Column 51
  - ... and 1 more occurrences

- **symbol** (appears 23 times):
  - Line 641, Column 48
  - Line 655, Column 51
  - Line 667, Column 51
  - Line 92, Column 80
  - Line 97, Column 16
  - ... and 18 more occurrences

- **test_consolidation_manager** (appears 1 times):
  - Line 715, Column 16

- **timeframe** (appears 19 times):
  - Line 641, Column 77
  - Line 655, Column 80
  - Line 667, Column 80
  - Line 92, Column 101
  - Line 97, Column 45
  - ... and 14 more occurrences

- **x** (appears 3 times):
  - Line 514, Column 43
  - Line 562, Column 46
  - Line 586, Column 43

---

### 43. src/training/model_interpretability/model_explainer.py

**Total Issues:** 171

**Issues by Name:**

- **Any** (appears 16 times):
  - Line 65, Column 15
  - Line 75, Column 19
  - Line 212, Column 15
  - Line 217, Column 19
  - Line 301, Column 77
  - ... and 11 more occurrences

- **Dict** (appears 14 times):
  - Line 75, Column 9
  - Line 217, Column 9
  - Line 301, Column 67
  - Line 553, Column 9
  - Line 622, Column 89
  - ... and 9 more occurrences

- **List** (appears 9 times):
  - Line 395, Column 9
  - Line 454, Column 9
  - Line 500, Column 9
  - Line 70, Column 23
  - Line 214, Column 23
  - ... and 4 more occurrences

- **X_test** (appears 4 times):
  - Line 92, Column 36
  - Line 111, Column 29
  - Line 130, Column 29
  - Line 577, Column 29

- **X_train** (appears 4 times):
  - Line 91, Column 40
  - Line 148, Column 26
  - Line 110, Column 30
  - Line 576, Column 30

- **e** (appears 16 times):
  - Line 56, Column 90
  - Line 203, Column 77
  - Line 204, Column 66
  - Line 294, Column 73
  - Line 295, Column 62
  - ... and 11 more occurrences

- **f** (appears 19 times):
  - Line 337, Column 38
  - Line 341, Column 34
  - Line 345, Column 35
  - Line 402, Column 38
  - Line 408, Column 37
  - ... and 14 more occurrences

- **feature** (appears 11 times):
  - Line 274, Column 50
  - Line 257, Column 19
  - Line 261, Column 19
  - Line 265, Column 19
  - Line 235, Column 54
  - ... and 6 more occurrences

- **feature_names** (appears 8 times):
  - Line 253, Column 27
  - Line 90, Column 37
  - Line 193, Column 53
  - Line 234, Column 44
  - Line 149, Column 32
  - ... and 3 more occurrences

- **features** (appears 2 times):
  - Line 651, Column 40
  - Line 655, Column 90

- **i** (appears 1 times):
  - Line 235, Column 88

- **individual_results** (appears 4 times):
  - Line 642, Column 39
  - Line 696, Column 27
  - Line 634, Column 31
  - Line 691, Column 31

- **k** (appears 1 times):
  - Line 661, Column 39

- **keyword** (appears 9 times):
  - Line 337, Column 69
  - Line 341, Column 65
  - Line 345, Column 66
  - Line 402, Column 69
  - Line 408, Column 68
  - ... and 4 more occurrences

- **log_call** (appears 9 times):
  - Line 61, Column 5
  - Line 208, Column 5
  - Line 299, Column 5
  - Line 388, Column 5
  - Line 448, Column 5
  - ... and 4 more occurrences

- **model** (appears 6 times):
  - Line 232, Column 23
  - Line 233, Column 35
  - Line 147, Column 24
  - Line 109, Column 28
  - Line 129, Column 28
  - ... and 1 more occurrences

- **model_name** (appears 11 times):
  - Line 86, Column 30
  - Line 77, Column 78
  - Line 78, Column 68
  - Line 587, Column 46
  - Line 645, Column 33
  - ... and 6 more occurrences

- **models** (appears 4 times):
  - Line 559, Column 35
  - Line 570, Column 37
  - Line 555, Column 88
  - Line 556, Column 78

- **output_dir** (appears 7 times):
  - Line 81, Column 25
  - Line 604, Column 30
  - Line 114, Column 33
  - Line 133, Column 33
  - Line 167, Column 33
  - ... and 2 more occurrences

- **traced** (appears 9 times):
  - Line 62, Column 5
  - Line 209, Column 5
  - Line 300, Column 5
  - Line 389, Column 5
  - Line 449, Column 5
  - ... and 4 more occurrences

- **v** (appears 2 times):
  - Line 661, Column 42
  - Line 661, Column 87

- **validates** (appears 1 times):
  - Line 60, Column 5

- **x** (appears 2 times):
  - Line 272, Column 78
  - Line 663, Column 90

- **y_test** (appears 1 times):
  - Line 579, Column 29

- **y_train** (appears 1 times):
  - Line 578, Column 30

---

### 44. research/clusters/advanced_feature_engineering.py

**Total Issues:** 171

**Issues by Name:**

- **AdvancedMarkovFeatureEngine** (appears 1 times):
  - Line 1252, Column 21

- **col** (appears 12 times):
  - Line 1269, Column 22
  - Line 1023, Column 32
  - Line 1133, Column 39
  - Line 1024, Column 25
  - Line 1029, Column 36
  - ... and 7 more occurrences

- **col_data** (appears 1 times):
  - Line 231, Column 56

- **col_name** (appears 1 times):
  - Line 231, Column 42

- **column** (appears 2 times):
  - Line 1107, Column 12
  - Line 1108, Column 34

- **feature** (appears 2 times):
  - Line 1141, Column 40
  - Line 1141, Column 60

- **horizon** (appears 101 times):
  - Line 280, Column 81
  - Line 285, Column 27
  - Line 293, Column 89
  - Line 302, Column 96
  - Line 312, Column 71
  - ... and 96 more occurrences

- **i** (appears 45 times):
  - Line 1222, Column 11
  - Line 1230, Column 15
  - Line 509, Column 22
  - Line 541, Column 22
  - Line 593, Column 22
  - ... and 40 more occurrences

- **j** (appears 2 times):
  - Line 891, Column 33
  - Line 891, Column 53

- **model** (appears 4 times):
  - Line 1197, Column 36
  - Line 1198, Column 48
  - Line 1199, Column 34
  - Line 1200, Column 28

---

### 45. src/launcher/ares_launcher.py

**Total Issues:** 170

**Issues by Name:**

- **candidate** (appears 3 times):
  - Line 1024, Column 12
  - Line 1025, Column 15
  - Line 1025, Column 54

- **col** (appears 1 times):
  - Line 393, Column 61

- **dep** (appears 4 times):
  - Line 1015, Column 19
  - Line 1017, Column 44
  - Line 1018, Column 19
  - Line 1019, Column 47

- **e** (appears 12 times):
  - Line 54, Column 64
  - Line 1160, Column 39
  - Line 1541, Column 74
  - Line 2385, Column 71
  - Line 2386, Column 46
  - ... and 7 more occurrences

- **f** (appears 79 times):
  - Line 311, Column 36
  - Line 1659, Column 33
  - Line 2294, Column 38
  - Line 576, Column 41
  - Line 332, Column 16
  - ... and 74 more occurrences

- **finfo** (appears 3 times):
  - Line 419, Column 54
  - Line 420, Column 55
  - Line 421, Column 52

- **fname** (appears 1 times):
  - Line 422, Column 69

- **i** (appears 11 times):
  - Line 1359, Column 15
  - Line 1396, Column 45
  - Line 1423, Column 23
  - Line 1936, Column 25
  - Line 1359, Column 65
  - ... and 6 more occurrences

- **k** (appears 8 times):
  - Line 742, Column 27
  - Line 784, Column 27
  - Line 849, Column 27
  - Line 742, Column 67
  - Line 784, Column 67
  - ... and 3 more occurrences

- **key** (appears 1 times):
  - Line 466, Column 46

- **main** (appears 1 times):
  - Line 2390, Column 16

- **output_file** (appears 1 times):
  - Line 474, Column 42

- **pipeline** (appears 1 times):
  - Line 2279, Column 32

- **pipeline_type** (appears 5 times):
  - Line 1351, Column 11
  - Line 1366, Column 36
  - Line 1349, Column 76
  - Line 1354, Column 55
  - Line 654, Column 59

- **pipelines** (appears 1 times):
  - Line 2278, Column 28

- **r** (appears 2 times):
  - Line 1132, Column 40
  - Line 1132, Column 67

- **stage_dependency_names** (appears 3 times):
  - Line 1015, Column 26
  - Line 1025, Column 28
  - Line 1019, Column 20

- **start_from_step** (appears 4 times):
  - Line 1382, Column 38
  - Line 1359, Column 20
  - Line 1363, Column 73
  - Line 654, Column 82

- **stop_at_step** (appears 6 times):
  - Line 1383, Column 11
  - Line 1363, Column 105
  - Line 1384, Column 39
  - Line 654, Column 99
  - Line 1359, Column 41
  - ... and 1 more occurrences

- **sub_result** (appears 13 times):
  - Line 1190, Column 27
  - Line 1261, Column 27
  - Line 1334, Column 27
  - Line 1128, Column 31
  - Line 1191, Column 95
  - ... and 8 more occurrences

- **tprint_error** (appears 1 times):
  - Line 1542, Column 12

- **v** (appears 4 times):
  - Line 742, Column 30
  - Line 784, Column 30
  - Line 849, Column 30
  - Line 973, Column 31

- **value** (appears 2 times):
  - Line 465, Column 46
  - Line 466, Column 53

- **visited_dependencies** (appears 2 times):
  - Line 1010, Column 23
  - Line 1012, Column 12

- **word** (appears 1 times):
  - Line 1505, Column 38

---

### 46. src/training/steps/data_collection/klines_downloading_processing.py

**Total Issues:** 170

**Issues by Name:**

- **api_key** (appears 6 times):
  - Line 284, Column 27
  - Line 1216, Column 16
  - Line 1492, Column 16
  - Line 1574, Column 16
  - Line 278, Column 19
  - ... and 1 more occurrences

- **api_secret** (appears 6 times):
  - Line 285, Column 30
  - Line 1217, Column 19
  - Line 1493, Column 19
  - Line 1575, Column 19
  - Line 278, Column 34
  - ... and 1 more occurrences

- **col** (appears 30 times):
  - Line 970, Column 24
  - Line 188, Column 29
  - Line 189, Column 26
  - Line 977, Column 15
  - Line 970, Column 56
  - ... and 25 more occurrences

- **col_name** (appears 1 times):
  - Line 473, Column 31

- **col_value** (appears 1 times):
  - Line 473, Column 43

- **combined_data** (appears 10 times):
  - Line 170, Column 19
  - Line 179, Column 40
  - Line 597, Column 19
  - Line 601, Column 32
  - Line 882, Column 19
  - ... and 5 more occurrences

- **combined_dtypes** (appears 9 times):
  - Line 798, Column 35
  - Line 796, Column 41
  - Line 808, Column 57
  - Line 812, Column 63
  - Line 797, Column 37
  - ... and 4 more occurrences

- **combined_nulls** (appears 5 times):
  - Line 799, Column 37
  - Line 808, Column 74
  - Line 786, Column 38
  - Line 787, Column 28
  - Line 789, Column 28

- **count** (appears 2 times):
  - Line 787, Column 50
  - Line 789, Column 51

- **create_consolidated** (appears 4 times):
  - Line 1219, Column 28
  - Line 1495, Column 28
  - Line 1577, Column 28
  - Line 309, Column 36

- **date_ranges** (appears 4 times):
  - Line 801, Column 15
  - Line 803, Column 33
  - Line 804, Column 31
  - Line 774, Column 20

- **dtype** (appears 3 times):
  - Line 779, Column 55
  - Line 780, Column 57
  - Line 781, Column 115

- **e** (appears 27 times):
  - Line 229, Column 29
  - Line 414, Column 29
  - Line 507, Column 29
  - Line 561, Column 71
  - Line 654, Column 29
  - ... and 22 more occurrences

- **error** (appears 2 times):
  - Line 1519, Column 27
  - Line 1692, Column 35

- **file_path** (appears 19 times):
  - Line 142, Column 41
  - Line 372, Column 41
  - Line 448, Column 41
  - Line 592, Column 41
  - Line 770, Column 41
  - ... and 14 more occurrences

- **group** (appears 2 times):
  - Line 615, Column 23
  - Line 618, Column 51

- **input_file** (appears 5 times):
  - Line 1391, Column 33
  - Line 1403, Column 30
  - Line 1388, Column 36
  - Line 1423, Column 30
  - Line 1420, Column 45

- **issue** (appears 3 times):
  - Line 1339, Column 31
  - Line 1143, Column 37
  - Line 1139, Column 48

- **main** (appears 1 times):
  - Line 1695, Column 20

- **max_gap_minutes** (appears 8 times):
  - Line 536, Column 67
  - Line 1218, Column 24
  - Line 1485, Column 37
  - Line 1494, Column 24
  - Line 1576, Column 24
  - ... and 3 more occurrences

- **null_count** (appears 4 times):
  - Line 985, Column 15
  - Line 986, Column 27
  - Line 989, Column 21
  - Line 990, Column 54

- **parquet_file** (appears 1 times):
  - Line 843, Column 29

- **rec** (appears 2 times):
  - Line 1164, Column 31
  - Line 1345, Column 31

- **resampling_intervals** (appears 3 times):
  - Line 293, Column 15
  - Line 1220, Column 29
  - Line 295, Column 37

- **run_ethusdt_3year_pipeline** (appears 1 times):
  - Line 1679, Column 28

- **step** (appears 1 times):
  - Line 1514, Column 27

- **t** (appears 2 times):
  - Line 995, Column 23
  - Line 979, Column 27

- **warning** (appears 2 times):
  - Line 1526, Column 27
  - Line 1687, Column 35

- **warnings** (appears 3 times):
  - Line 639, Column 28
  - Line 623, Column 16
  - Line 629, Column 16

- **x** (appears 3 times):
  - Line 1020, Column 21
  - Line 846, Column 37
  - Line 1022, Column 21

---

### 47. research/cluster_analysis/market_factor_analysis/factor_extraction.py

**Total Issues:** 170

**Issues by Name:**

- **col** (appears 12 times):
  - Line 1269, Column 22
  - Line 1023, Column 32
  - Line 1133, Column 39
  - Line 1024, Column 25
  - Line 1029, Column 36
  - ... and 7 more occurrences

- **col_data** (appears 1 times):
  - Line 231, Column 56

- **col_name** (appears 1 times):
  - Line 231, Column 42

- **column** (appears 2 times):
  - Line 1107, Column 12
  - Line 1108, Column 34

- **feature** (appears 2 times):
  - Line 1141, Column 40
  - Line 1141, Column 60

- **horizon** (appears 101 times):
  - Line 280, Column 81
  - Line 285, Column 27
  - Line 293, Column 89
  - Line 302, Column 96
  - Line 312, Column 71
  - ... and 96 more occurrences

- **i** (appears 45 times):
  - Line 1222, Column 11
  - Line 1230, Column 15
  - Line 509, Column 22
  - Line 541, Column 22
  - Line 593, Column 22
  - ... and 40 more occurrences

- **j** (appears 2 times):
  - Line 891, Column 33
  - Line 891, Column 53

- **model** (appears 4 times):
  - Line 1197, Column 36
  - Line 1198, Column 48
  - Line 1199, Column 34
  - Line 1200, Column 28

---

### 48. src/training/steps/data_collection/data_preparation/validate_and_fix_aggtrades_format.py

**Total Issues:** 168

**Issues by Name:**

- **args** (appears 1 times):
  - Line 1186, Column 37

- **c** (appears 2 times):
  - Line 542, Column 33
  - Line 542, Column 68

- **col** (appears 65 times):
  - Line 377, Column 15
  - Line 194, Column 19
  - Line 346, Column 15
  - Line 529, Column 19
  - Line 563, Column 19
  - ... and 60 more occurrences

- **e** (appears 31 times):
  - Line 1190, Column 37
  - Line 1196, Column 26
  - Line 1373, Column 82
  - Line 1384, Column 31
  - Line 236, Column 59
  - ... and 26 more occurrences

- **error_recovery** (appears 2 times):
  - Line 1339, Column 11
  - Line 1340, Column 27

- **expected_dtype** (appears 12 times):
  - Line 379, Column 35
  - Line 195, Column 45
  - Line 740, Column 39
  - Line 567, Column 27
  - Line 380, Column 113
  - ... and 7 more occurrences

- **feature** (appears 2 times):
  - Line 323, Column 15
  - Line 324, Column 67

- **files** (appears 2 times):
  - Line 934, Column 25
  - Line 925, Column 12

- **func** (appears 1 times):
  - Line 1186, Column 31

- **i** (appears 9 times):
  - Line 227, Column 48
  - Line 228, Column 46
  - Line 229, Column 46
  - Line 230, Column 46
  - Line 1223, Column 58
  - ... and 4 more occurrences

- **issue** (appears 1 times):
  - Line 1066, Column 48

- **k** (appears 2 times):
  - Line 510, Column 42
  - Line 510, Column 85

- **kwargs** (appears 1 times):
  - Line 1186, Column 45

- **memory_monitor** (appears 8 times):
  - Line 1333, Column 11
  - Line 1352, Column 73
  - Line 1333, Column 30
  - Line 1334, Column 12
  - Line 1335, Column 15
  - ... and 3 more occurrences

- **process_single_file** (appears 1 times):
  - Line 925, Column 19

- **process_task** (appears 1 times):
  - Line 1215, Column 14

- **processor_func** (appears 1 times):
  - Line 1211, Column 65

- **stability_metrics** (appears 6 times):
  - Line 1328, Column 7
  - Line 1355, Column 11
  - Line 1367, Column 11
  - Line 1329, Column 8
  - Line 1356, Column 12
  - ... and 1 more occurrences

- **task** (appears 2 times):
  - Line 1211, Column 81
  - Line 1215, Column 27

- **tasks** (appears 1 times):
  - Line 1215, Column 45

- **traced** (appears 7 times):
  - Line 1302, Column 1
  - Line 122, Column 5
  - Line 135, Column 5
  - Line 384, Column 5
  - Line 794, Column 5
  - ... and 2 more occurrences

- **use_concurrency** (appears 2 times):
  - Line 825, Column 15
  - Line 849, Column 41

- **v** (appears 2 times):
  - Line 510, Column 45
  - Line 510, Column 105

- **validate_file_with_stability** (appears 2 times):
  - Line 895, Column 42
  - Line 942, Column 42

- **validates** (appears 1 times):
  - Line 134, Column 5

- **validator** (appears 2 times):
  - Line 1344, Column 21
  - Line 1341, Column 16

- **x** (appears 1 times):
  - Line 1296, Column 34

---

### 49. research/clusters/economic_metrics.py

**Total Issues:** 167

**Issues by Name:**

- **category** (appears 1 times):
  - Line 1893, Column 36

- **change_idx** (appears 10 times):
  - Line 1423, Column 15
  - Line 1423, Column 36
  - Line 1425, Column 56
  - Line 1426, Column 62
  - Line 1429, Column 44
  - ... and 5 more occurrences

- **data** (appears 11 times):
  - Line 770, Column 27
  - Line 860, Column 36
  - Line 958, Column 35
  - Line 1028, Column 34
  - Line 1120, Column 29
  - ... and 6 more occurrences

- **duration** (appears 1 times):
  - Line 1631, Column 36

- **e** (appears 1 times):
  - Line 222, Column 98

- **i** (appears 34 times):
  - Line 659, Column 29
  - Line 663, Column 47
  - Line 741, Column 31
  - Line 819, Column 31
  - Line 908, Column 31
  - ... and 29 more occurrences

- **k** (appears 11 times):
  - Line 788, Column 40
  - Line 878, Column 40
  - Line 976, Column 40
  - Line 1046, Column 40
  - Line 1138, Column 40
  - ... and 6 more occurrences

- **logger** (appears 1 times):
  - Line 222, Column 12

- **m** (appears 2 times):
  - Line 1891, Column 51
  - Line 1891, Column 74

- **metrics** (appears 1 times):
  - Line 1891, Column 63

- **profile** (appears 2 times):
  - Line 567, Column 28
  - Line 586, Column 44

- **reg** (appears 1 times):
  - Line 1631, Column 86

- **regime** (appears 59 times):
  - Line 325, Column 24
  - Line 326, Column 23
  - Line 173, Column 44
  - Line 174, Column 27
  - Line 428, Column 44
  - ... and 54 more occurrences

- **result** (appears 16 times):
  - Line 1846, Column 12
  - Line 1847, Column 15
  - Line 1901, Column 23
  - Line 1907, Column 23
  - Line 1897, Column 38
  - ... and 11 more occurrences

- **t** (appears 4 times):
  - Line 1449, Column 44
  - Line 1450, Column 41
  - Line 1451, Column 38
  - Line 1452, Column 43

- **v** (appears 11 times):
  - Line 788, Column 44
  - Line 878, Column 44
  - Line 976, Column 44
  - Line 1046, Column 44
  - Line 1138, Column 44
  - ... and 6 more occurrences

- **value** (appears 1 times):
  - Line 1910, Column 69

---

### 50. src/training/steps/models_training/analyst_models_training.py

**Total Issues:** 166

**Issues by Name:**

- **all_oof_predictions** (appears 2 times):
  - Line 358, Column 73
  - Line 378, Column 28

- **base_name** (appears 1 times):
  - Line 378, Column 48

- **dataframe_sources** (appears 2 times):
  - Line 1067, Column 29
  - Line 1060, Column 16

- **dict_sources** (appears 2 times):
  - Line 1064, Column 30
  - Line 1055, Column 16

- **direction_models_metrics** (appears 3 times):
  - Line 1732, Column 15
  - Line 1733, Column 46
  - Line 1724, Column 20

- **e** (appears 28 times):
  - Line 49, Column 63
  - Line 61, Column 74
  - Line 73, Column 71
  - Line 84, Column 93
  - Line 94, Column 82
  - ... and 23 more occurrences

- **feature_map** (appears 6 times):
  - Line 1111, Column 15
  - Line 1094, Column 8
  - Line 1103, Column 8
  - Line 1108, Column 8
  - Line 1082, Column 20
  - ... and 1 more occurrences

- **fold** (appears 1 times):
  - Line 623, Column 73

- **formatted_predictions** (appears 3 times):
  - Line 989, Column 63
  - Line 1000, Column 22
  - Line 1006, Column 41

- **frame** (appears 2 times):
  - Line 1068, Column 31
  - Line 1069, Column 31

- **i** (appears 3 times):
  - Line 810, Column 47
  - Line 1082, Column 47
  - Line 1082, Column 105

- **key** (appears 2 times):
  - Line 1053, Column 31
  - Line 1058, Column 38

- **kwargs** (appears 56 times):
  - Line 1775, Column 73
  - Line 321, Column 33
  - Line 322, Column 25
  - Line 944, Column 37
  - Line 965, Column 18
  - ... and 51 more occurrences

- **model_info** (appears 18 times):
  - Line 1228, Column 43
  - Line 1355, Column 43
  - Line 1723, Column 43
  - Line 1725, Column 41
  - Line 1726, Column 39
  - ... and 13 more occurrences

- **model_name** (appears 4 times):
  - Line 1227, Column 57
  - Line 1354, Column 57
  - Line 1722, Column 73
  - Line 371, Column 69

- **mt** (appears 2 times):
  - Line 399, Column 40
  - Line 1382, Column 32

- **name** (appears 5 times):
  - Line 1068, Column 23
  - Line 1065, Column 23
  - Line 1066, Column 38
  - Line 1065, Column 49
  - Line 1069, Column 37

- **names** (appears 1 times):
  - Line 1063, Column 24

- **regime_id** (appears 5 times):
  - Line 1227, Column 45
  - Line 1234, Column 34
  - Line 1354, Column 45
  - Line 1361, Column 34
  - Line 1722, Column 61

- **regime_models** (appears 3 times):
  - Line 1226, Column 46
  - Line 1353, Column 46
  - Line 1721, Column 46

- **source** (appears 3 times):
  - Line 1065, Column 31
  - Line 1066, Column 31
  - Line 1065, Column 42

- **train_idx** (appears 6 times):
  - Line 627, Column 40
  - Line 625, Column 35
  - Line 626, Column 35
  - Line 710, Column 64
  - Line 712, Column 28
  - ... and 1 more occurrences

- **val_idx** (appears 8 times):
  - Line 653, Column 28
  - Line 654, Column 28
  - Line 655, Column 28
  - Line 719, Column 32
  - Line 625, Column 49
  - ... and 3 more occurrences

---

## Remaining Files (1480 files)

| File | Issues |
|------|--------|
| src/training/steps/market_analysis/multi_horizon_sub_pipeline_adapter.py | 163 |
| research/cluster_analysis/economic_relevance/trading_significance.py | 163 |
| src/training/steps/market_analysis/nas_regime/core/enhanced_perfect_nas_regime_detector.py | 162 |
| src/training/steps/pre_training/unified_data_driven_pipeline/statistical_analysis/statistical_framework.py | 160 |
| src/utils/ml_common/optimization/pareto.py | 159 |
| src/utils/common_operations.py | 158 |
| src/utils/data/cli.py | 156 |
| research/cluster_analysis/market_factor_analysis/dimension_discovery.py | 155 |
| research/clusters/dimension_analyzer.py | 155 |
| src/feature_generation/utils/cross_timeframe_analysis_pipeline.py | 153 |
| src/training/steps/data_collection/data_preparation/enhanced_data_quality_manager.py | 152 |
| src/training/steps/data_collection/klines_downloading_processing_enhanced.py | 151 |
| research/clusters/comprehensive_feature_integration.py | 151 |
| src/feature_generation/utils/enhanced_sr_feature_extractor.py | 150 |
| src/monitoring/enhanced_ml_monitoring.py | 148 |
| src/research/crypto_analysis/automated_crypto_processor.py | 146 |
| research/crypto_analysis/automated_crypto_processor.py | 146 |
| src/training/steps/backtesting/abc_testing/results_visualization.py | 144 |
| src/training/steps/pre_training/unified_data_driven_pipeline/feature_selection/multi_objective_selector.py | 144 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/advanced_feature_selection.py | 142 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/feature_engineering_pipeline.py | 142 |
| src/utils/vectorbt_batch_processor.py | 142 |
| src/research/crypto_analysis/data_analyzer.py | 140 |
| research/crypto_analysis/data_analyzer.py | 140 |
| src/feature_generation/utils/unified_vectorization_manager.py | 139 |
| src/utils/ml_common/optimization/shared_utils/evolutionary_search.py | 138 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/hyperparameter_optimization.py | 135 |
| src/utils/matrix_operations/unified_operations.py | 135 |
| src/training/steps/market_analysis/clusters/step8_validation.py | 134 |
| src/trading/execution/exchange_interface.py | 133 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/enhanced_statistical_framework.py | 133 |
| src/training/model_interpretability/shap_analyzer.py | 133 |
| exchanges/base_exchange/base_exchange.py | 133 |
| src/utils/parquet_utils.py | 132 |
| research/clusters/automated_feature_engineering.py | 132 |
| exchanges/bingx_production.py | 132 |
| src/monitoring/trading_integration.py | 130 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/unified_performance_monitor.py | 130 |
| src/utils/ml_common/validation/enhanced_overfitting_detection.py | 129 |
| src/monitoring/daily_summary_tracker.py | 128 |
| src/research/profit_labeling/contextual_feature_labeling.py | 128 |
| src/training/steps/model_training/analyst_ensemble_training.py | 128 |
| src/monitoring/gui/enhanced_dashboard.py | 127 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/search_strategies.py | 127 |
| src/training/utils/feature_selection/data_validation.py | 126 |
| exchanges/bingx_fixed.py | 126 |
| exchanges/shared/klines_downloading_processing.py | 126 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/enhanced_feature_generator.py | 125 |
| src/core/decorators/validate.py | 125 |
| src/training/steps/pre_training/unified_data_driven_pipeline/core/vectorbt_optimizer.py | 124 |
| src/training/steps/market_analysis/components/nas_tas_regime_discovery.py | 124 |
| src/explainability/explainability_orchestrator.py | 123 |
| src/training/steps/model_training/sub_pipeline.py | 123 |
| src/training/utils/feature_selection/partial_information_decompositor.py | 123 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/detailed_pipeline_reporter.py | 122 |
| src/utils/ml_common/ensembles/ensemble_manager.py | 122 |
| src/monitoring/gui/monitoring_dashboard.py | 121 |
| src/feature_generation/categories/oscillator.py | 120 |
| src/explainability/visualization_tools.py | 120 |
| src/nas_tas/data/data_processor.py | 120 |
| src/monitoring/enhanced_monitoring_orchestrator.py | 119 |
| src/training/steps/data_collection/data_preparation/comprehensive_gap_filler.py | 119 |
| src/training/steps/backtesting/real_parameters_optimization.py | 119 |
| src/training/steps/market_analysis/components/regime_ensemble_training.py | 119 |
| src/utils/ml_common/explainability/model_interpretability.py | 116 |
| src/training/steps/market_analysis/tas_regime/regime_analysis/unsupervised_regime_detection.py | 114 |
| src/training/steps/model_training/model_validation.py | 114 |
| src/utils/sr_clustering/parameter_optimization_engine.py | 114 |
| src/utils/ml_common/utils/lookahead_protection.py | 114 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/lightgbm_featuretools_generator.py | 113 |
| research/cluster_analysis/market_factor_analysis/feature_clustering.py | 113 |
| src/training/steps/data_collection/validators/pipeline_validators.py | 112 |
| src/training/steps/backtesting/real_reporting_engine.py | 112 |
| src/monitoring/ensemble_monitor.py | 111 |
| src/trading/integration/data_integration.py | 111 |
| src/feature_generation/utils/optimization/unified_optimizer.py | 111 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/htf_template_system.py | 111 |
| src/training/model_interpretability/interpretability_visualizer.py | 111 |
| src/analyst/unified_regime_classifier_sr_focused.py | 111 |
| src/training/steps/market_analysis/clusters/step10_comprehensive_reporting.py | 110 |
| src/utils/ml_common/confidence_metrics.py | 110 |
| exchanges/bingx.py | 110 |
| exchanges/phemex.py | 109 |
| src/feature_generation/utils/multi_timeframe_training_analysis.py | 107 |
| src/training/steps/data_collection/sub_pipeline.py | 107 |
| src/utils/model_performance_monitor.py | 107 |
| src/training/steps/market_analysis/tas_regime/data_pipeline/data_validation.py | 106 |
| src/research/price_patterns/run_complete_pattern_discovery.py | 105 |
| src/training/steps/market_analysis/components/hybrid_nas_tas_regime_discovery.py | 105 |
| src/utils/matrix_operations/vectorbt_optimizations.py | 105 |
| research/price_patterns/run_complete_pattern_discovery.py | 105 |
| research/clusters/production_feature_integration.py | 104 |
| src/supervisor/dynamic_weighter.py | 102 |
| src/feature_engineering_roadmap/transforms.py | 102 |
| src/training/steps/market_analysis/optimal_regime_clustering_backup/enhanced_clustering_improvements.py | 102 |
| src/research/price_patterns/run_pure_pattern_discovery.py | 101 |
| src/explainability/sr_explainer.py | 101 |
| src/tactician/ml_tactics_manager.py | 101 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/evolutionary_algorithms.py | 101 |
| src/training/steps/model_training/tactician_training_step.py | 101 |
| research/price_patterns/run_pure_pattern_discovery.py | 101 |
| research/vectorbt_optimizations/feature_comparison_optimizer.py | 101 |
| src/monitoring/shap_lime_integration.py | 100 |
| src/utils/ml_common/validation/enhanced_overfitting_detection_with_learning_curves.py | 100 |
| src/training/steps/data_collection/data_preparation_components/data_integrity_checker.py | 99 |
| src/training/steps/data_collection/utils/data_operations_utils.py | 99 |
| src/utils/ml_common/models/enhanced_model_trainer.py | 99 |
| src/utils/ml_common/training/vectorized_training_manager.py | 99 |
| data_quality/mapping/data_flow.py | 99 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/unified_search_algorithms.py | 98 |
| research/profit_labeling/contextual_feature_labeling.py | 98 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/model_validation.py | 97 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/unified_multi_objective_optimizer.py | 97 |
| src/training/steps/market_analysis/clusters/features/analyzer.py | 97 |
| src/training/utils/feature_selection/main_framework.py | 97 |
| exchanges/shared/high_level_wrappers_typed.py | 97 |
| src/monitoring/explainability_integration.py | 96 |
| src/feature_generation/core/feature_bank.py | 96 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/core/economic_clustering.py | 96 |
| src/training/model_interpretability/interpretability_reporter.py | 96 |
| src/utils/data/quality/data_cleaning.py | 96 |
| src/analyst/meta_labeling_system.py | 96 |
| src/trading/signal_generation/signal_pipeline.py | 95 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/shared_validation.py | 95 |
| src/training/steps/models_training/analyst_ensemble_training.py | 94 |
| src/training/utils/feature_selection/selection_methods.py | 94 |
| src/utils/sr_clustering/weight_optimization_engine.py | 94 |
| src/research/price_patterns/pattern_discovery_framework.py | 93 |
| src/training/steps/market_analysis/optimized_process_engines.py | 93 |
| src/utils/ml_common/optimization/hierarchical_hpo.py | 93 |
| research/price_patterns/pattern_discovery_framework.py | 93 |
| src/explainability/integration_decorators.py | 91 |
| src/training/steps/market_analysis/clusters/step9_results_consolidation.py | 91 |
| src/utils/data/feature_engineer.py | 91 |
| research/candle_based_features/interpretability_analysis.py | 91 |
| research/candle_ml_patterns/interpretability_analysis.py | 91 |
| src/training/steps/market_analysis/optimal_regime_clustering_backup/metrics_evolution_report.py | 90 |
| live_trading/config_manager.py | 90 |
| src/monitoring/trade_decision_capture.py | 89 |
| src/feature_generation/utils/step06_enhanced_feature_engineering_step.py | 89 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/feature_collection.py | 89 |
| src/training/model_interpretability/lime_analyzer.py | 89 |
| src/feature_generation/utils/step06_enhanced_feature_engineering.py | 88 |
| src/training/steps/data_collection/data_preparation_components/quality_metrics_calculator.py | 88 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/randomforest_feature_generator.py | 88 |
| src/training/steps/pre_training/unified_data_driven_pipeline/core/template_interaction_generator.py | 88 |
| src/feature_selection/advanced/enhanced_multi_stage_rfe.py | 87 |
| src/training/steps/market_analysis/sub_pipeline.py | 87 |
| src/utils/ml_common/evaluation/unified_evaluator.py | 87 |
| src/utils/ml_common/validation/cv.py | 87 |
| src/feature_generation/categories/returns.py | 86 |
| src/training/steps/models_training/tactician_ensemble_training.py | 86 |
| src/training/steps/market_analysis/tas_regime/components/micro_regime_detector.py | 86 |
| src/training/utils/feature_selection/temporal_analysis.py | 86 |
| research/clusters/visualization.py | 86 |
| exchanges/gateio.py | 86 |
| src/training/steps/data_collection/data_preparation/missing_data_downloader_and_gap_filler.py | 85 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/advanced_lookback_optimizer.py | 85 |
| src/utils/ml_common/unified_vectorization_manager.py | 85 |
| src/utils/ml_common/utils/memory_integration.py | 85 |
| src/utils/data/klines_parquet.py | 85 |
| src/utils/config/security.py | 85 |
| src/database/precomputed_features_manager.py | 85 |
| src/trading/execution/live_trader.py | 84 |
| src/training/steps/data_collection/decorators/step_decorators.py | 84 |
| src/training/steps/backtesting/sub_pipeline.py | 84 |
| src/training/steps/market_analysis/coverage_constrained_clustering/clusterer.py | 84 |
| src/training/utils/feature_selection/stability_analysis.py | 84 |
| src/utils/validation/unified_framework.py | 84 |
| src/utils/hardware/m1_gpu_utils.py | 84 |
| src/analyst/enhanced_regime_predictor.py | 84 |
| src/analyst/data_utils.py | 84 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/common_feature_logic.py | 83 |
| src/training/steps/market_analysis/tas_regime/search/multi_objective_search.py | 83 |
| src/core/decorators/auth.py | 83 |
| src/trading/regime/regime_analyzer.py | 82 |
| src/feature_generation/core/feature_generator.py | 82 |
| src/feature_generation/categories/momentum.py | 81 |
| src/research/profit_labeling/dynamic_target_optimizer.py | 81 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/performance_estimators.py | 81 |
| src/training/utils/feature_selection partial_information_decomposition.py | 81 |
| src/utils/ml_common/models/multi_output_models.py | 81 |
| src/utils/matrix_operations/vectorized_core.py | 81 |
| research/profit_labeling/dynamic_target_optimizer.py | 81 |
| src/feature_generation/matrix_integration/matrix_processor.py | 80 |
| src/training/steps/data_collection/data_preparation/step02_5_financial_logging.py | 80 |
| live_trading/data_streamer.py | 80 |
| src/monitoring/monitoring_orchestrator.py | 79 |
| src/tactician/async_order_executor.py | 79 |
| src/training/steps/pre_training/unified_data_driven_pipeline/core/intelligent_feature_selector.py | 79 |
| src/training/steps/market_analysis/optimal_regime_clustering_backup/clustering.py | 79 |
| src/training/steps/market_analysis/model_persistence_components/model_persistence_step.py | 79 |
| data_quality/unified_quality_orchestrator.py | 79 |
| src/trading/monitoring/performance_tracker.py | 78 |
| src/trading/execution/paper_trader.py | 78 |
| src/feature_generation/core/vectorbt_feature_generator.py | 77 |
| src/research/price_patterns/pure_price_action_patterns.py | 77 |
| src/training/steps/data_collection/enhanced_append_data_downloader.py | 77 |
| src/training/steps/backtesting/nas_tas_deprecated/walk_forward_analyzer.py | 77 |
| src/training/steps/market_analysis/nas_regime/core/enhanced_search_strategies.py | 77 |
| src/features_common/transforms/vectorbt_scaler.py | 77 |
| src/utils/sr_clustering/predictive_sr_engine.py | 77 |
| src/analyst/unified_regime_classifier_fractal_simplified.py | 77 |
| research/price_patterns/pure_price_action_patterns.py | 77 |
| research/cluster_analysis/price_patterns/pure_price_patterns.py | 77 |
| src/research/mixed_factor_analysis/ml_pattern_discovery.py | 76 |
| src/training/steps/data_collection/enhanced_api_agnostic_data_collector.py | 76 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/unified_clustering_algorithms.py | 76 |
| src/analyst/location_classifier_improvements.py | 76 |
| research/mixed_factor_analysis/ml_pattern_discovery.py | 76 |
| research/cluster_analysis/price_patterns/ml_discovery/anomaly_discovery.py | 76 |
| src/interfaces/enhanced_event_bus.py | 75 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/math_validation_integration.py | 75 |
| src/utils/ml_common/validation/validation_utils.py | 75 |
| data_quality/mapping/dead_code.py | 75 |
| src/feature_engineering_roadmap/disagreement_meta_features.py | 74 |
| src/training/steps/market_analysis/regime_handler.py | 74 |
| src/utils/sr_clustering/trading_ml_integration.py | 74 |
| src/utils/ml_common/utils/base_safeguards.py | 74 |
| src/utils/data/gap_detector.py | 74 |
| src/nas_tas/unified_pipeline.py | 74 |
| src/research/profit_labeling/parameter_optimizer.py | 73 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/core/coherent_regime_modeling.py | 73 |
| research/profit_labeling/parameter_optimizer.py | 73 |
| exchanges/order_router.py | 73 |
| exchanges/trading_receiver.py | 73 |
| src/research/profit_labeling/ensemble_labeling_system.py | 72 |
| src/research/mixed_factor_analysis/economic_relevance_research_framework.py | 72 |
| src/tactician/tactics_orchestrator.py | 72 |
| src/training/steps/main_training_pipeline.py | 72 |
| src/training/steps/market_analysis/tas_regime/adaptation/dynamic_optimization.py | 72 |
| src/training/steps/market_analysis/clusters/step1_feature_preparation.py | 72 |
| src/utils/async_utils.py | 72 |
| research/profit_labeling/ensemble_labeling_system.py | 72 |
| research/mixed_factor_analysis/economic_relevance_research_framework.py | 72 |
| research/cluster_analysis/economic_relevance/causal_analysis.py | 72 |
| exchanges/shared/high_level_wrappers.py | 72 |
| src/explainability/tactician_explainer.py | 71 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/common_lookback_optimizer.py | 71 |
| src/core/domain/decorators_extended.py | 71 |
| research/clusters/trading_calibration.py | 71 |
| src/supervisor/enhanced_prediction_service.py | 70 |
| src/training/steps/market_analysis/shared_utils/features.py | 70 |
| src/utils/report_manager.py | 70 |
| src/utils/hardware/adaptive_optimization_engine.py | 70 |
| src/analyst/unified_regime_classifier_fractal_enhanced.py | 70 |
| src/research/price_patterns/ml_pure_price_pattern_discovery.py | 69 |
| src/training/steps/data_collection/data_quality_components/anomaly_detector.py | 69 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/advanced_data_loading.py | 69 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/multi_objective_optimizer.py | 69 |
| src/analyst/candlestick_pattern_analyzer.py | 69 |
| live_trading/trading_engine.py | 69 |
| research/price_patterns/ml_pure_price_pattern_discovery.py | 69 |
| research/cluster_analysis/price_patterns/ml_discovery/clustering_discovery.py | 69 |
| research/vectorbt_optimizations/price_patterns_optimizer.py | 69 |
| src/research/crypto_analysis/optimized_crypto_processor.py | 68 |
| src/training/steps/market_analysis/model_persistence_components/metadata_tracker.py | 68 |
| research/crypto_analysis/optimized_crypto_processor.py | 68 |
| src/trading/integration/training_integration.py | 67 |
| src/supervisor/performance_reporter.py | 67 |
| src/feature_generation/categories/vectorbt_acceleration.py | 67 |
| src/feature_selection/advanced/native_validation.py | 67 |
| src/utils/enhanced_mlflow_integration.py | 67 |
| research/cluster_analysis/economic_relevance/market_state_relevance.py | 67 |
| research/clusters/dimension_economic_relevance.py | 67 |
| data_quality/mapping/call_graph.py | 67 |
| src/trading/utils/helpers.py | 66 |
| src/research/price_patterns/matrix_profile_discovery.py | 66 |
| src/training/steps/market_analysis/optimal_regime_clustering_backup/enhanced_optimized_clustering.py | 66 |
| research/candle_based_features/ml_neural_indicators.py | 66 |
| research/candle_ml_patterns/ml_neural_indicators.py | 66 |
| research/feature_comparison/feature_acceleration_dilation_enhanced.py | 66 |
| research/price_patterns/matrix_profile_discovery.py | 66 |
| research/cluster_analysis/price_patterns/ml_discovery/matrix_profile_discovery.py | 66 |
| research/cluster_analysis/clustering/validation_metrics.py | 66 |
| research/clusters/validation_metrics.py | 66 |
| src/launcher/validation_utilities.py | 65 |
| src/training/steps/pre_training/tactician_entry_labeler.py | 65 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/comprehensive_validator.py | 65 |
| src/utils/nas_tas/core/tas_engine.py | 65 |
| research/feature_comparison/multi_target_system.py | 65 |
| research/vectorbt_optimizations/clustering_optimizer.py | 65 |
| exchanges/base_exchange/response_handler.py | 65 |
| src/supervisor/main.py | 64 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/enhanced_hybrid_orchestrator.py | 64 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/core/hybrid_regime_detector.py | 64 |
| src/training/steps/market_analysis/clusters/feature_service.py | 64 |
| src/utils/tprint.py | 64 |
| src/utils/hardware/m1_cpu_optimizer.py | 64 |
| research/vectorbt_optimizations/profit_labeling_optimizer.py | 64 |
| src/training/steps/backtesting/abc_testing/statistical_analysis.py | 63 |
| src/utils/data/historical_data_pipeline.py | 63 |
| src/analyst/feature_engineering_utils.py | 63 |
| research/feature_comparison/standardized_features.py | 63 |
| research/cluster_analysis/clustering/regime_discovery.py | 63 |
| research/clusters/regime_clusterer.py | 63 |
| src/trading/monitoring/regime_monitor.py | 62 |
| src/feature_generation/categories/volatility.py | 62 |
| src/feature_generation/utils/fractional_differentiation_pipeline.py | 62 |
| src/feature_generation/utils/enhanced_matrix_accelerator.py | 62 |
| src/research/profit_labeling/backtesting_integrated_validator.py | 62 |
| src/research/profit_labeling/advanced_statistical_validator.py | 62 |
| src/research/price_patterns/advanced_pattern_definitions.py | 62 |
| src/explainability/hmm_explainer.py | 62 |
| src/training/steps/pre_training/analyst_profit_labeler.py | 62 |
| src/utils/ml_common/data_processing/data_labeling.py | 62 |
| src/utils/common_ml/backtesting/model_saver.py | 62 |
| research/candle_based_features/ml_candle_pattern_indicators.py | 62 |
| research/candle_based_features/enhanced_consensus_system.py | 62 |
| research/candle_ml_patterns/ml_candle_pattern_indicators.py | 62 |
| research/candle_ml_patterns/enhanced_consensus_system.py | 62 |
| research/profit_labeling/advanced_statistical_validator.py | 62 |
| research/price_patterns/advanced_pattern_definitions.py | 62 |
| research/clusters/feature_importance.py | 62 |
| GUI/api_server_simple.py | 62 |
| src/feature_generation/categories/microstructure_features.py | 61 |
| src/training/steps/data_collection/data_preparation_components/data_cleaner.py | 61 |
| src/training/steps/market_analysis/optimal_regime_clustering_backup/utils.py | 61 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/core/feature_engine_integration.py | 61 |
| src/training/steps/market_analysis/regime_analysis/label_fusion.py | 61 |
| src/utils/validator_orchestrator.py | 61 |
| src/utils/ml_common/optimization/shared_utils/feature_engineering.py | 61 |
| src/utils/ml_common/validation/data_leakage_detector.py | 61 |
| research/feature_comparison/enhanced_comparison_runner.py | 61 |
| src/research/mixed_factor_analysis/microstructure_impact_research.py | 60 |
| src/tactician/position_sizer.py | 60 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/enhanced_schema_validation.py | 60 |
| src/training/steps/market_analysis/regime_data_splitting/nas_tas_regime_data_splitting.py | 60 |
| src/utils/ml_common/validation/stability.py | 60 |
| research/mixed_factor_analysis/microstructure_impact_research.py | 60 |
| src/research/profit_labeling/bonus_penalty_optimizer.py | 59 |
| src/research/profit_labeling/labeling_visualizer.py | 59 |
| src/research/mixed_factor_analysis/volatility_impact_research.py | 59 |
| src/training/steps/data_collection/data_preparation/data_quality_dashboard.py | 59 |
| src/training/steps/backtesting/abc_testing/multi_model_orchestrator.py | 59 |
| src/training/steps/backtesting/abc_testing/performance_monitoring.py | 59 |
| src/training/steps/models_training/corrected_ml_entry_timing_labeler.py | 59 |
| src/training/steps/market_analysis/clusters/data_validator.py | 59 |
| src/training/steps/market_analysis/clusters/optimizer.py | 59 |
| src/utils/model_manager.py | 59 |
| src/utils/ml_common/training/enhanced_training_utils.py | 59 |
| src/nas_tas/error_handling.py | 59 |
| src/analyst/predictive_ensembles/ensemble_orchestrator.py | 59 |
| src/database/sqlite_manager.py | 59 |
| research/profit_labeling/labeling_visualizer.py | 59 |
| research/feature_comparison/analyst_labeler_integration.py | 59 |
| research/mixed_factor_analysis/volatility_impact_research.py | 59 |
| src/feature_selection/advanced/validation_framework.py | 58 |
| src/training/steps/backtesting/real_monte_carlo_engine.py | 58 |
| src/training/steps/market_analysis/tas_regime/search/evolutionary_search.py | 58 |
| src/utils/decorator_registry.py | 58 |
| src/nas_tas/training/training_orchestrator.py | 58 |
| exchanges/shared/unified_exchange_interface.py | 58 |
| src/trading/utils/validation.py | 57 |
| src/feature_generation/core/vectorbt_optimization_mixin.py | 57 |
| src/feature_generation/utils/vectorization_optimizer.py | 57 |
| src/feature_generation/utils/step06_utility_container.py | 57 |
| src/feature_generation/utils/optimization_validator.py | 57 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/advanced_performance_monitoring.py | 57 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/enhanced_data_integration.py | 57 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/unified_regime_analyzer.py | 57 |
| src/training/steps/market_analysis/components/component_factory.py | 57 |
| src/training/steps/market_analysis/nas_clustering/core/essential_nas_clusterer.py | 57 |
| src/training/steps/market_analysis/nas_regime/core/nas_search.py | 57 |
| src/training/steps/market_analysis/nas_regime/core/enhanced_matrix_operations.py | 57 |
| src/core/config_service.py | 57 |
| src/analyst/enhanced_prediction_integrator.py | 57 |
| research/candle_based_features/model_comparison_pipeline.py | 57 |
| research/candle_ml_patterns/model_comparison_pipeline.py | 57 |
| research/profit_labeling/bonus_penalty_optimizer.py | 57 |
| research/feature_comparison/stability_metrics.py | 57 |
| research/clusters/lookahead_bias_prevention.py | 57 |
| exchanges/shared/high_level_wrappers_typed_part2.py | 57 |
| src/training/steps/backtesting/unified_config.py | 56 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/unified_architecture_compression.py | 56 |
| src/utils/ml_common/ensembles/enhanced_oof_stacking_with_confidence.py | 56 |
| src/analyst/regime_expert_orchestrator.py | 56 |
| src/monitoring/regime_monitoring_dashboard.py | 55 |
| src/feature_generation/utils/data_driven_feature_selector.py | 55 |
| src/research/profit_labeling/labeling_validator.py | 55 |
| src/research/mixed_factor_analysis/pattern_ml_integration.py | 55 |
| src/feature_selection/advanced/enhanced_ensemble_selector.py | 55 |
| src/training/steps/standardized_parquet_handler.py | 55 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/core/financial_performance_metrics.py | 55 |
| src/training/steps/market_analysis/tas_regime/adaptation/performance_tracking.py | 55 |
| src/training/utils/feature_selection/performance_monitoring.py | 55 |
| src/training/simplified_architecture/enhanced_pipeline_orchestrator.py | 55 |
| src/utils/fallback_monitoring.py | 55 |
| src/utils/parallel_processing_optimizer.py | 55 |
| src/utils/data/processing/data_processing.py | 55 |
| src/utils/data/processing/transformers.py | 55 |
| research/profit_labeling/labeling_validator.py | 55 |
| research/mixed_factor_analysis/pattern_ml_integration.py | 55 |
| research/cluster_analysis/economic_relevance/pattern_dimension_analysis.py | 55 |
| src/monitoring/fractional_system_monitor.py | 54 |
| src/training/steps/pre_training/unified_data_driven_pipeline/core/modular_architecture.py | 54 |
| src/training/steps/market_analysis/sr_detection.py | 54 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/unified_evaluation_framework.py | 54 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/unified_architecture_config.py | 54 |
| src/utils/common_utilities.py | 54 |
| src/utils/ml_common/models/model_training.py | 54 |
| src/utils/ml_common/optimization/grid_utils.py | 54 |
| src/utils/data/historical_data_downloader.py | 54 |
| src/analyst/multi_timeframe_feature_engineering.py | 54 |
| src/monitoring/regime_performance_tracker.py | 53 |
| src/training/steps/pre_training/unified_data_driven_pipeline/core/config.py | 53 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/search_spaces.py | 53 |
| src/training/steps/market_analysis/clusters/performance_monitor.py | 53 |
| src/core/decorators.py | 53 |
| src/utils/ml_common/pipeline_orchestrator.py | 53 |
| src/utils/ml_common/ensembles/vectorbt_ensemble_optimizer.py | 53 |
| src/utils/matrix_operations/error_handling.py | 53 |
| src/feature_generation/categories/autoencoder.py | 52 |
| src/tactician/enhanced_scenario_based_predictor.py | 52 |
| src/feature_selection/vectorbt/vectorbt_feature_selector.py | 52 |
| src/training/steps/backtesting/abc_testing/configuration_management.py | 52 |
| src/training/steps/market_analysis/enhanced_validation_framework.py | 52 |
| src/training/steps/market_analysis/tas_regime/backtesting/risk_analysis.py | 52 |
| src/training/steps/market_analysis/model_persistence_components/model_registry.py | 52 |
| src/training/steps/market_analysis/clusters/optimization_service.py | 52 |
| src/features_common/transforms/categorical_encoding.py | 52 |
| src/utils/ml_common/optimization/enhanced_hpo_monitor.py | 52 |
| src/utils/ml_common/utils/data_quality.py | 52 |
| src/database/efficient_features_database.py | 52 |
| research/clusters/constraints.py | 52 |
| exchanges/shared/monitoring_dashboard.py | 52 |
| src/trading/execution/order_manager.py | 51 |
| src/feature_selection/dimensionality/vif_module.py | 51 |
| src/training/steps/market_analysis/clusters/metrics.py | 51 |
| src/core/decorators/logging.py | 51 |
| src/utils/decorators.py | 51 |
| src/utils/regime_aware_financial_logging_decorator.py | 51 |
| src/utils/ml_common/post_training/model_persistence.py | 51 |
| src/utils/ml_common/utils/memory_optimization.py | 51 |
| src/utils/ml_common/validation/model_complexity_analysis.py | 51 |
| src/utils/nas_tas/core/nas_engine.py | 51 |
| research/feature_comparison/compute_aware_optimizer.py | 51 |
| research/clusters/ml_integration_framework.py | 51 |
| exchanges/base_exchange/message_handler.py | 51 |
| src/feature_generation/utils/vectorbt_memory_optimizer.py | 50 |
| src/research/profit_labeling/real_time_monitor.py | 50 |
| src/research/profit_labeling/adaptive_labeling_strategy.py | 50 |
| src/feature_selection/vectorbt/vectorbt_mrmr_selector.py | 50 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/architecture_encoders.py | 50 |
| src/training/steps/market_analysis/model_persistence_components/model_serializer.py | 50 |
| src/training/utils/feature_selection/causal_analysis.py | 50 |
| src/features_common/transforms/scaling_normalization.py | 50 |
| src/utils/ml_common/reporting/enhanced_reporting_system.py | 50 |
| src/utils/ml_common/explainability/model_explanations.py | 50 |
| research/profit_labeling/real_time_monitor.py | 50 |
| research/profit_labeling/adaptive_labeling_strategy.py | 50 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/unified_economic_evaluator.py | 49 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/shared_metrics.py | 49 |
| src/training/steps/market_analysis/tas_regime/adaptation/real_time_adaptation.py | 49 |
| src/core/decorators/retry_timeout.py | 49 |
| src/utils/comprehensive_function_logger.py | 49 |
| src/utils/ml_common/optimization/regime_specific_tpsl_optimizer.py | 49 |
| src/utils/ml_common/data_processing/sr_feature_integration.py | 49 |
| src/analyst/predictive_ensembles.py | 49 |
| research/feature_comparison/comparison_report.py | 49 |
| research/clusters/integration_layer.py | 49 |
| data_quality/simple_quality_orchestrator.py | 49 |
| exchanges/shared/performance_monitor.py | 49 |
| src/trading/integration/exchange_integration.py | 48 |
| src/feature_generation/utils/optimized_cross_timeframe_analysis_integration.py | 48 |
| src/training/steps/data_collection/data_preparation/data_quality_monitor.py | 48 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/advanced_caching.py | 48 |
| src/training/steps/market_analysis/cluster_constraints.py | 48 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/enhanced_economic_evaluator.py | 48 |
| src/training/steps/market_analysis/tas_regime/search/rl_search.py | 48 |
| src/training/steps/market_analysis/nas_regime/core/enhanced_ml_common_integration.py | 48 |
| src/utils/regime_probability_analyzer.py | 48 |
| src/utils/ml_common/post_training/model_validation.py | 48 |
| src/utils/ml_common/validation/data_leakage_prevention.py | 48 |
| src/nas_tas/results/result_manager.py | 48 |
| src/analyst/sr_relevance_optimizer.py | 48 |
| research/feature_comparison/relevance_analyzer.py | 48 |
| research/clusters/dimension_discovery_pipeline.py | 48 |
| src/monitoring/gui/data_visualization.py | 47 |
| src/feature_selection/advanced/multi_stage_rfe.py | 47 |
| src/feature_selection/advanced/enhanced_advanced_selector.py | 47 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/clustering_quality_metrics.py | 47 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/data_normalization.py | 47 |
| src/training/utils/debug_utilities.py | 47 |
| src/utils/function_call_monitor.py | 47 |
| src/utils/ml_common/ensembles/stacking_confidence_calibration.py | 47 |
| research/vectorbt_optimizations/crypto_analysis_optimizer.py | 47 |
| data_quality/mapping/cli.py | 47 |
| exchanges/exchange_registry.py | 47 |
| src/trading/signal_generation/analyst_signals.py | 46 |
| src/trading/execution/trading_orchestrator.py | 46 |
| src/feature_generation/categories/spectral_features.py | 46 |
| src/feature_generation/core/vectorbt_batch_processor.py | 46 |
| src/training/steps/market_analysis/components/tas_regime_discovery.py | 46 |
| src/utils/ml_common/training/per_regime_training_step.py | 46 |
| src/utils/ml_common/validation/unified_cv.py | 46 |
| src/utils/data/basic_returns_engineer.py | 46 |
| src/utils/matrix_operations/computation_toolbox.py | 46 |
| research/feature_comparison/diagnostics.py | 46 |
| research/clusters/ml_enhanced_discovery.py | 46 |
| src/trading/monitoring/trade_monitor.py | 45 |
| src/trading/data/data_validator.py | 45 |
| src/explainability/analyst_explainer.py | 45 |
| src/training/steps/data_collection/data_quality_components/validation_decorators.py | 45 |
| src/training/steps/data_collection/data_preparation_components/data_format_converter.py | 45 |
| src/training/utils/feature_selection/base_framework.py | 45 |
| src/utils/kline_parquet.py | 45 |
| src/utils/ml_common/training/ensemble_training_step.py | 45 |
| src/utils/matrix_operations/batch_operations.py | 45 |
| research/feature_comparison/pre_screening_pipeline.py | 45 |
| research/cluster_analysis/clustering/similarity_clustering.py | 45 |
| research/clusters/similarity_matrix_clustering.py | 45 |
| exchanges/data_aggregator.py | 45 |
| src/components/modular_supervisor.py | 44 |
| src/tactician/tactician.py | 44 |
| src/feature_selection/vectorbt/vectorbt_memory_optimizer.py | 44 |
| src/training/steps/data_collection/monitoring/pipeline_monitor.py | 44 |
| src/training/steps/backtesting/nas_tas_deprecated/performance_attribution.py | 44 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/advanced_search_strategies.py | 44 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/core/performance_estimator.py | 44 |
| src/training/steps/market_analysis/tas_regime/search/advanced_search.py | 44 |
| src/training/simplified_architecture/enhanced_config_system.py | 44 |
| src/core/sr_error_handlers.py | 44 |
| src/core/domain/decorators.py | 44 |
| src/core/decorators/cache.py | 44 |
| src/utils/ml_common/training/training_integration.py | 44 |
| src/nas_tas/evaluation/unified_evaluator.py | 44 |
| research/clusters/empirical_threshold_discovery.py | 44 |
| src/trading/signal_generation/tactician_signals.py | 43 |
| src/trading/integration/model_integration.py | 43 |
| src/supervisor/performance_monitor.py | 43 |
| src/custom_types/validation.py | 43 |
| src/tactician/sr_levels/sr_modules/sr_metrics_calculator.py | 43 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/regime_alignment_manager.py | 43 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/analysis_components.py | 43 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/core/architecture_encoder.py | 43 |
| src/training/steps/market_analysis/monitoring/performance_monitor.py | 43 |
| src/training/steps/market_analysis/tas_regime/utils/tree_utils.py | 43 |
| src/training/steps/market_analysis/regime_data_splitting/validator.py | 43 |
| src/core/decorators/enhanced_error_handling.py | 43 |
| src/utils/mlflow_utils.py | 43 |
| src/analyst/predictive_ensembles/multi_timeframe_ensemble.py | 43 |
| exchanges/shared/enhanced_unified_exchange_interface.py | 43 |
| src/models/stacker_lgbm_gate.py | 42 |
| src/trading/signal_generation/signal_combiner.py | 42 |
| src/feature_selection/advanced/prefiltering.py | 42 |
| src/training/common/artifact_persistence.py | 42 |
| src/training/steps/data_collection/data_preparation/gap_filler_pipeline.py | 42 |
| src/training/steps/data_collection/data_quality_components/validation_strategies.py | 42 |
| src/training/steps/market_analysis/monitoring/function_call_monitor.py | 42 |
| src/training/steps/market_analysis/clusters/weighted_category_pca.py | 42 |
| src/utils/ml_common/optimization/tree_architecture_search.py | 42 |
| src/utils/ml_common/optimization/specialized_trading_trees.py | 42 |
| live_trading/order_manager.py | 42 |
| live_trading/trading_orchestrator.py | 42 |
| research/clusters/enhanced_price_action_analysis.py | 42 |
| data_quality/generate_unified_report.py | 42 |
| exchanges/shared/market/instrument_manager.py | 42 |
| src/trading/regime/regime_classifier.py | 41 |
| src/feature_generation/utils/enhanced_data_driven_interaction_generator.py | 41 |
| src/research/profit_labeling/heuristic_analyzer.py | 41 |
| src/tactician/scenario_based_predictor.py | 41 |
| src/feature_selection/vectorbt/vectorbt_mutual_information.py | 41 |
| src/training/steps/models_training/analyst_training_pipeline.py | 41 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/advanced_validation.py | 41 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/unified_search_space_evolution.py | 41 |
| src/training/steps/market_analysis/components/nas_regime_discovery.py | 41 |
| src/training/steps/market_analysis/clusters/clustering_utils.py | 41 |
| src/features_common/mixins/vectorbt_mixin.py | 41 |
| src/utils/ml_common/utils/enhanced_error_handling.py | 41 |
| src/utils/ml_common/validation/hpo_overfitting_prevention.py | 41 |
| research/profit_labeling/heuristic_analyzer.py | 41 |
| research/feature_comparison/feature_consolidation.py | 41 |
| exchanges/shared/auth/subaccount_manager.py | 41 |
| src/components/modular_analyst.py | 40 |
| src/trading/sizing/position_sizer.py | 40 |
| src/feature_generation/categories/advanced_statistical.py | 40 |
| src/feature_generation/categories/time.py | 40 |
| src/feature_generation/utils/optimization_config.py | 40 |
| src/training/steps/data_collection/data_preparation/step1_orchestrator.py | 40 |
| src/training/steps/data_collection/utils/common_operations.py | 40 |
| src/training/steps/models_training/analyst_pre_ml_orchestration.py | 40 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/core/financial_optimizers.py | 40 |
| src/training/steps/market_analysis/tas_regime/trading/trading_engine.py | 40 |
| src/training/steps/market_analysis/nas_clustering/core/evaluation/multi_objective.py | 40 |
| src/training/steps/market_analysis/nas_regime/core/hybrid_architecture.py | 40 |
| src/core/decorators/function_monitor.py | 40 |
| src/features_common/transforms/base_scaler.py | 40 |
| src/analyst/meta_label_relevance.py | 40 |
| research/feature_comparison/feature_scorecard.py | 40 |
| src/trading/config/regime_config.py | 39 |
| src/feature_generation/utils/vectorbt_performance_benchmark.py | 39 |
| src/feature_selection/vectorbt/vectorbt_correlation_filter.py | 39 |
| src/training/steps/data_collection/data_preparation_components/training_validation_config.py | 39 |
| src/training/steps/models_training/tactician_training_pipeline.py | 39 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/evaluation/economic_evaluator.py | 39 |
| src/training/steps/market_analysis/components/artifact_manager.py | 39 |
| src/training/simplified_architecture/dependency_injection.py | 39 |
| src/features_common/mixins/optimization_mixin.py | 39 |
| src/utils/enhanced_data_quality_validator.py | 39 |
| src/utils/financial_metrics_logger.py | 39 |
| src/utils/ml_common/evaluation/enhanced_bootstrap_confidence_intervals.py | 39 |
| src/utils/data/quality/advanced_quality_metrics.py | 39 |
| GUI/launcher_integration.py | 39 |
| src/research/profit_labeling/ml_label_quality_assessor.py | 38 |
| src/training/steps/feature_engineering/price_action/close_location_value.py | 38 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/economic_evaluation.py | 38 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/core/regime_aware_search.py | 38 |
| src/utils/enhanced_financial_metrics_logger.py | 38 |
| src/utils/enhanced_step_optimizations.py | 38 |
| src/utils/validation.py | 38 |
| src/utils/ml_common/optimization/adaptive_regime_nas.py | 38 |
| src/utils/ml_common/optimization/neural_architecture_search.py | 38 |
| src/utils/ml_common/data_processing/feature_preparation.py | 38 |
| src/strategist/strategist.py | 38 |
| research/profit_labeling/ml_label_quality_assessor.py | 38 |
| research/feature_comparison/run_comparison.py | 38 |
| exchanges/shared/config_manager.py | 38 |
| src/components/modular_tactician.py | 37 |
| src/training/steps/data_collection/unified_gap_filler.py | 37 |
| src/training/steps/feature_engineering/trend/trend_coherence.py | 37 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/advanced_error_handling.py | 37 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/unified_trading_viability_evaluator.py | 37 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/core/nas_financial_optimizer.py | 37 |
| src/training/steps/market_analysis/tas_regime/components/advanced_tree_models.py | 37 |
| src/training/steps/market_analysis/nas_regime/core/advanced_neural_architectures.py | 37 |
| src/utils/data_access_protection.py | 37 |
| src/utils/pipeline_standards.py | 37 |
| src/utils/ml_common/vectorbt_portfolio_optimization.py | 37 |
| src/utils/ml_common/post_training/model_evaluation.py | 37 |
| src/utils/data/monthly_data_downloader.py | 37 |
| src/utils/data/quality/data_qualification_imports.py | 37 |
| src/validation/regime_consensus_validator.py | 37 |
| live_trading/unified_trading_system.py | 37 |
| research/candle_based_features/ml_indicator_training_pipeline.py | 37 |
| research/candle_ml_patterns/ml_indicator_training_pipeline.py | 37 |
| research/clusters/adaptive_clustering.py | 37 |
| exchanges/shared/unified_ohlcv_standardizer.py | 37 |
| exchanges/shared/unified_exchange_standardizer.py | 37 |
| src/ci/validators.py | 36 |
| src/components/modular_strategist.py | 36 |
| src/trading/regime/regime_weights.py | 36 |
| src/research/profit_labeling/enhanced_multi_horizon_labeler.py | 36 |
| src/research/price_patterns/pattern_discovery_example.py | 36 |
| src/feature_engineering_roadmap/assembly_dag.py | 36 |
| src/tactician/comprehensive_enhanced_scenario_predictor.py | 36 |
| src/training/steps/backtesting/vectorbt_unified_manager.py | 36 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/modular_architecture.py | 36 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/shared_training.py | 36 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/core/financial_search_strategies.py | 36 |
| src/training/steps/market_analysis/optimized_multi_horizon_optimizer/grid_bayesian_optimizer.py | 36 |
| src/training/steps/market_analysis/tas_regime/search/bayesian_search.py | 36 |
| src/config/sr_config_loader.py | 36 |
| src/utils/matrix_operations.py | 36 |
| src/utils/ml_common/monitoring/enhanced_error_detector.py | 36 |
| src/utils/hmm/hardware_integration.py | 36 |
| src/utils/nas_tas/optimization/architecture_search.py | 36 |
| research/candle_based_features/consensus_indicator_system.py | 36 |
| research/candle_ml_patterns/consensus_indicator_system.py | 36 |
| research/profit_labeling/enhanced_multi_horizon_labeler.py | 36 |
| src/trading/data/market_data_provider.py | 35 |
| src/explainability/base_explainer.py | 35 |
| src/tactician/step17_optimized_tactician.py | 35 |
| src/feature_selection/analysis/feature_importance_analyzer.py | 35 |
| src/feature_selection/parallel/parallel_feature_selector.py | 35 |
| src/training/steps/backtesting/nas_tas_deprecated/validation_orchestrator.py | 35 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/advanced_artifact_management.py | 35 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/evaluation/clustering_cross_validation.py | 35 |
| src/training/core/decorators.py | 35 |
| src/training/utils/feature_selection/quality_metrics.py | 35 |
| src/utils/enhanced_step_wrapper.py | 35 |
| src/utils/ml_common/validation/overfitting_monitoring.py | 35 |
| src/utils/data/quality/quality_alert_system.py | 35 |
| research/feature_comparison/enhanced_relevance_analyzer.py | 35 |
| exchanges/exchange_dispatcher.py | 35 |
| exchanges/shared/pricing/enhanced_ohlcv_manager.py | 35 |
| src/monitoring/surrogate_optimization_monitor.py | 34 |
| src/launcher/enhanced_trading_launcher.py | 34 |
| src/trading/monitoring/alert_manager.py | 34 |
| src/feature_generation/categories/entropy.py | 34 |
| src/feature_generation/utils/step06_labeling_components/optimized_triple_barrier_labeling_improved.py | 34 |
| src/research/price_patterns/core_patterns.py | 34 |
| src/feature_selection/vectorbt/vectorbt_unified_framework.py | 34 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/clustering_quality_analyzer.py | 34 |
| src/utils/ml_common/data_drift_detector.py | 34 |
| src/utils/matrix_operations/hardware_integration.py | 34 |
| src/utils/nas_tas/optimization/strategy_search.py | 34 |
| src/utils/hardware/m1_memory_optimizer.py | 34 |
| src/utils/common_ml/backtesting/analytics_reporter.py | 34 |
| research/feature_comparison/feature_acceleration_dilation.py | 34 |
| research/price_patterns/core_patterns.py | 34 |
| research/cluster_analysis/price_patterns/mathematical_definitions.py | 34 |
| research/clusters/refined_ml_discovery.py | 34 |
| src/feature_generation/utils/limited_microstructure_features.py | 33 |
| src/feature_engineering_roadmap/data_contracts.py | 33 |
| src/tactician/position_division_strategy.py | 33 |
| src/feature_selection/dimensionality/pca_module.py | 33 |
| src/feature_selection/advanced/improved_mrmr.py | 33 |
| src/feature_selection/chunked/chunked_processor.py | 33 |
| src/training/steps/data_collection/enhanced_data_validation_framework.py | 33 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/consensus_validator.py | 33 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/unified_hardware_manager.py | 33 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/unified_validation_system.py | 33 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/automatic_training/regime_training_pipeline.py | 33 |
| src/training/steps/market_analysis/model_persistence_components/version_manager.py | 33 |
| src/training/steps/market_analysis/clusters/hardware_service.py | 33 |
| src/training/steps/market_analysis/nas_regime/core/nas_shared_utils_integration.py | 33 |
| src/core/decorators/trace.py | 33 |
| src/utils/performance_utils.py | 33 |
| src/utils/ml_common/optimization/regime_hpo_wrapper.py | 33 |
| src/utils/data/optimized_parquet_storage.py | 33 |
| src/database/migration_utils.py | 33 |
| exchanges/shared/tests/verify_type_coverage.py | 33 |
| src/feature_generation/utils/enhanced_matrix_operations.py | 32 |
| src/feature_generation/utils/optimized_feature_pipeline.py | 32 |
| src/tactician/sr_levels/sr_modules/sr_feature_extractor.py | 32 |
| src/feature_selection/error_handling/enhanced_error_handler.py | 32 |
| src/feature_selection/advanced/advanced_selector.py | 32 |
| src/feature_selection/vectorbt/vectorbt_rfe_selector.py | 32 |
| src/training/steps/market_analysis/logging_standards.py | 32 |
| src/training/steps/market_analysis/shared_utils/metrics.py | 32 |
| src/training/steps/market_analysis/tas_regime/tree_cvlSA_demo.py | 32 |
| src/training/steps/market_analysis/tas_regime/backtesting/data_manager.py | 32 |
| src/features_common/vectorbt/unified_manager.py | 32 |
| src/utils/core/file_operations.py | 32 |
| src/utils/hardware/unified_hardware_manager.py | 32 |
| src/strategist/enhanced_regime_classifier.py | 32 |
| src/validation/walkforward_validation.py | 32 |
| live_trading/risk_manager.py | 32 |
| src/models/patch_gru.py | 31 |
| src/trading/execution/paper_trading_integration.py | 31 |
| src/feature_generation/utils/math_validation.py | 31 |
| src/research/profit_labeling/research_runner.py | 31 |
| src/research/price_patterns/lstm_discovery.py | 31 |
| src/integration/paper_trading_integration.py | 31 |
| src/nas_tas_integration/unified_regime_training_pipeline.py | 31 |
| src/training/steps/data_collection/data_quality_components/data_utils.py | 31 |
| src/training/steps/backtesting/abc_testing/risk_management.py | 31 |
| src/training/steps/feature_engineering/volatility/atr_volatility_ratio.py | 31 |
| src/training/steps/market_analysis/optimization_monitor.py | 31 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/data_caching.py | 31 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/core/advanced_search_strategies.py | 31 |
| src/training/steps/market_analysis/shared_utils/feature_filters.py | 31 |
| src/training/steps/market_analysis/nas_regime/core/enhanced_nas_integration.py | 31 |
| src/training/steps/model_training/random_survival_forest_tactician.py | 31 |
| src/training/steps/model_training/bayesian_optimization_msm.py | 31 |
| src/training/simplified_architecture/config_driven_architecture.py | 31 |
| src/core/domain/__init__.py | 31 |
| src/utils/ml_common/utils/thread_guard.py | 31 |
| src/utils/ml_common/validation/model_enhancement_guide.py | 31 |
| live_trading/api_client.py | 31 |
| research/profit_labeling/research_runner.py | 31 |
| research/price_patterns/lstm_discovery.py | 31 |
| research/cluster_analysis/price_patterns/ml_discovery/lstm_discovery.py | 31 |
| research/clusters/metric_orthogonalization.py | 31 |
| src/models/enhanced_patchtst.py | 30 |
| src/models/vectorbt_enhanced_models.py | 30 |
| src/trading/execution/live_trading_scheduler.py | 30 |
| src/trading/sizing/risk_calculator.py | 30 |
| src/research/profit_labeling/bonus_penalty_integration_example.py | 30 |
| src/research/profit_labeling/example_usage.py | 30 |
| src/feature_selection/memory/memory_efficient_selector.py | 30 |
| src/training/steps/data_collection/data_preparation/data_gap_detector.py | 30 |
| src/training/steps/data_collection/data_preparation_components/aggtrades_data_formatting.py | 30 |
| src/training/steps/models_training/ml_based_entry_timing_labeler.py | 30 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/feature_bank_integration.py | 30 |
| src/training/steps/market_analysis/tas_regime/core/tas_engine.py | 30 |
| src/training/steps/market_analysis/clusters/clustering_service.py | 30 |
| src/features_common/mixins/caching_mixin.py | 30 |
| src/utils/error_handler.py | 30 |
| src/utils/unified_cache.py | 30 |
| src/utils/import_standardizer.py | 30 |
| research/feature_comparison/family_diverse_features.py | 30 |
| src/trading/regime/regime_detector.py | 29 |
| src/feature_generation/categories/candlestick_pattern.py | 29 |
| src/research/profit_labeling/enhanced_example_usage.py | 29 |
| src/research/price_patterns/gradient_targets.py | 29 |
| src/feature_selection/vectorbt/vectorbt_regularization.py | 29 |
| src/training/steps/data_collection/data_download_monitor.py | 29 |
| src/training/steps/feature_engineering/price_action/bar_efficiency_ratio.py | 29 |
| src/training/steps/pre_training/standardized_labeling_interface.py | 29 |
| src/training/steps/pre_training/artifacts/manifest.py | 29 |
| src/training/steps/pre_training/unified_data_driven_pipeline/steps/feature_generation_data_validation_step.py | 29 |
| src/training/steps/pre_training/unified_data_driven_pipeline/core/economic_evaluator.py | 29 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/data_pipeline.py | 29 |
| src/training/steps/market_analysis/tas_regime/meta_learning/tree_meta_learning.py | 29 |
| src/training/steps/market_analysis/clustering/main_component.py | 29 |
| src/utils/trading_decorators.py | 29 |
| src/utils/ml_common/models/model_registry.py | 29 |
| src/utils/ml_common/validation/cv_utils.py | 29 |
| src/utils/ml_common/ensembles/oof_stacking_ensemble_manager.py | 29 |
| src/utils/hardware/advanced_cpu_optimizer.py | 29 |
| src/utils/hardware/m1_optimizations.py | 29 |
| src/analyst/predictive_ensembles/regime_ensembles/volatile_regime_ensemble.py | 29 |
| research/candle_based_features/ml_indicator_integration.py | 29 |
| research/candle_ml_patterns/ml_indicator_integration.py | 29 |
| research/price_patterns/gradient_targets.py | 29 |
| src/monitoring/fractional_performance_tracker.py | 28 |
| src/models/lgbm_gru_embedding.py | 28 |
| src/feature_generation/utils/step06_enhanced_validation_framework.py | 28 |
| src/feature_generation/utils/optimized_cross_timeframe_analysis.py | 28 |
| src/training/steps/data_collection/data_preparation/step02_data_reading.py | 28 |
| src/training/steps/models_training/negative_learning_training_patches.py | 28 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/enhanced_caching_integration.py | 28 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/enhanced_ml_integration.py | 28 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/regime_model_mapping/data_driven_model_selector.py | 28 |
| src/training/steps/market_analysis/regime_analysis/metrics.py | 28 |
| src/training/steps/market_analysis/tas_regime/backtesting/performance_attribution.py | 28 |
| src/training/steps/market_analysis/tas_regime/backtesting/monte_carlo.py | 28 |
| src/training/steps/market_analysis/tas_regime/evaluation/tas_evaluator.py | 28 |
| src/training/steps/market_analysis/regime_model_mapping/data_driven_model_selector.py | 28 |
| src/training/steps/market_analysis/clusters/memory_manager.py | 28 |
| src/training/steps/market_analysis/clusters/cv_enhancement_strategies.py | 28 |
| src/training/steps/market_analysis/clusters/step2_initial_clustering.py | 28 |
| src/features_common/mixins/validation_mixin.py | 28 |
| src/utils/decorators/errors.py | 28 |
| src/utils/ml_common/models/multiscale_nbeats.py | 28 |
| src/utils/ml_common/integration/enhanced_ml_pipeline_integration.py | 28 |
| src/utils/ml_common/training/universal_validation_integration.py | 28 |
| src/utils/ml_common/utils/logging_utils.py | 28 |
| src/utils/hardware/advanced_memory_optimizer.py | 28 |
| src/nas_tas/evaluation/performance_monitor.py | 28 |
| research/cluster_analysis/market_factor_analysis/statistical_analysis.py | 28 |
| research/clusters/statistical_dimension_analysis.py | 28 |
| src/monitoring/retrain_monitoring.py | 27 |
| src/feature_generation/test_tprint_logging.py | 27 |
| src/feature_generation/categories/representation_learning.py | 27 |
| src/feature_generation/core/factory.py | 27 |
| src/feature_generation/utils/centralized_logging.py | 27 |
| src/training/steps/data_collection/unified_data_loader.py | 27 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/unified_ensemble_search_space.py | 27 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/core/enhanced_economic_clustering.py | 27 |
| src/training/steps/market_analysis/monitoring/error_handler.py | 27 |
| src/training/steps/market_analysis/components/nas_ensemble_training.py | 27 |
| src/training/steps/market_analysis/tas_regime/data_pipeline/data_storage.py | 27 |
| src/training/steps/market_analysis/tas_regime/data_pipeline/pipeline_orchestrator.py | 27 |
| src/training/steps/market_analysis/nas_regime/core/enhanced_nas_modeling_integration.py | 27 |
| src/training/steps/market_analysis/nas_regime/core/perfect_nas_regime_detector.py | 27 |
| src/utils/enhanced_artifact_manager.py | 27 |
| src/utils/error_recovery/advanced_error_recovery.py | 27 |
| src/utils/data/quality/comprehensive_duplicate_analyzer.py | 27 |
| research/feature_comparison/time_series_validation.py | 27 |
| research/cluster_analysis/economic_relevance/__init__.py | 27 |
| src/monitoring/auto_monitoring_launcher.py | 26 |
| src/monitoring/trading_mode_monitoring_integration.py | 26 |
| src/common/config/loader.py | 26 |
| src/trading/data/live_data_collector.py | 26 |
| src/feature_generation/utils/vectorbt_optimization_integration.py | 26 |
| src/feature_selection/advanced/dynamic_selection.py | 26 |
| src/feature_selection/vectorbt/vectorbt_rolling_operations.py | 26 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/gpu_optimizations.py | 26 |
| src/training/steps/market_analysis/enhanced_validation.py | 26 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/position_aware_trading.py | 26 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/core/dynamic_search_space.py | 26 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/core/architecture_signal_generator.py | 26 |
| src/training/steps/market_analysis/shared_utils/feature_importance_pipeline_utils.py | 26 |
| src/training/steps/market_analysis/nas_clustering/core/micro_regime_detector.py | 26 |
| src/training/steps/market_analysis/nas_regime/validate_implementations.py | 26 |
| src/core/error_classes.py | 26 |
| src/utils/ml_common/integrated_analysis_pipeline.py | 26 |
| src/utils/ml_common/optimization/creative_tree_models.py | 26 |
| src/utils/ml_common/examples/universal_validation_demo.py | 26 |
| src/utils/ml_common/training/enhanced_early_stopping.py | 26 |
| src/utils/ml_common/ensembles/ensembling.py | 26 |
| src/utils/data/quality/data_qualification_error_handler.py | 26 |
| src/utils/matrix_operations/convenience.py | 26 |
| examples/partial_bar_nowcasting_demo.py | 26 |
| src/models/enhanced_tft.py | 25 |
| src/trading/execution/partial_bar_nowcasting.py | 25 |
| src/feature_generation/test_backward_compatibility.py | 25 |
| src/feature_generation/core/optimization_mixin.py | 25 |
| src/feature_generation/utils/unified_optimization_system.py | 25 |
| src/tactician/dynamic_barrier_calculator.py | 25 |
| src/tactician/ml_target_validator.py | 25 |
| src/feature_selection/advanced/confidence_scoring.py | 25 |
| src/training/steps/data_collection/data_collection_orchestrator.py | 25 |
| src/training/steps/data_collection/data_preparation/data_resampler.py | 25 |
| src/training/steps/market_analysis/regime_processing_decorator.py | 25 |
| src/training/steps/market_analysis/shared_utils/feature_importance_integration.py | 25 |
| src/training/steps/market_analysis/nas_clustering/core/nas_clusterer.py | 25 |
| src/training/steps/market_analysis/nas_modeling/core/advanced_preprocessing.py | 25 |
| src/training/steps/market_analysis/clusters/features/preprocessor.py | 25 |
| src/core/errors/base.py | 25 |
| src/utils/pipeline_enhancement_integration.py | 25 |
| src/utils/report_collector.py | 25 |
| src/utils/ml_common/training/quick_integration.py | 25 |
| src/utils/ml_common/training/base_training_step.py | 25 |
| src/utils/ml_common/validation/universal_temporal_validation.py | 25 |
| research/clusters/core_regime_discovery.py | 25 |
| exchanges/shared/monitoring_api.py | 25 |
| exchanges/shared/data_validation_suite.py | 25 |
| exchanges/shared/orders/order_manager.py | 25 |
| scripts/diagnose_regime_data_leakage.py | 24 |
| src/ares_pipeline.py | 24 |
| src/launcher/pipeline_managers.py | 24 |
| src/trading/sizing/leverage_manager.py | 24 |
| src/feature_generation/core/generator_factory.py | 24 |
| src/feature_generation/utils/optimization_metrics.py | 24 |
| src/feature_generation/utils/error_handling.py | 24 |
| src/feature_generation/utils/unified_optimization_wrapper.py | 24 |
| src/feature_selection/vectorbt/vectorbt_stability_selection.py | 24 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/constraint_systems.py | 24 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/core/advanced_clustering.py | 24 |
| src/training/steps/market_analysis/components/imports.py | 24 |
| src/training/steps/market_analysis/tas_regime/trading/signal_generator.py | 24 |
| src/training/steps/market_analysis/clusters/risk_mitigation.py | 24 |
| src/utils/input_validation.py | 24 |
| src/utils/step_validation_system.py | 24 |
| src/utils/ml_common/examples/automatic_validation_demo.py | 24 |
| research/feature_comparison/optimized_feature_versions.py | 24 |
| src/interfaces/event_bus.py | 23 |
| src/trading/utils/error_handling.py | 23 |
| src/supervisor/model_behavior_tracker.py | 23 |
| src/training/steps/data_collection/unified_resampler.py | 23 |
| src/training/steps/data_collection/utils/monitoring.py | 23 |
| src/training/steps/pre_training/unified_data_driven_pipeline/time_series_cv/purged_embargoed_cv.py | 23 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/enhanced_walk_forward_validation.py | 23 |
| src/training/steps/pre_training/unified_data_driven_pipeline/steps/feature_generation_feature_generation_step.py | 23 |
| src/training/steps/market_analysis/nas_tas_comparison_analysis.py | 23 |
| src/training/steps/market_analysis/optimal_regime_clustering_backup/enhanced_analysis.py | 23 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/core/multi_objective_optimizer.py | 23 |
| src/training/steps/market_analysis/tas_regime/components/neural_architecture.py | 23 |
| src/training/steps/market_analysis/tas_regime/data_pipeline/data_ingestion.py | 23 |
| src/training/steps/market_analysis/clusters/m1_optimizer.py | 23 |
| src/training/steps/market_analysis/clusters/gpu_manager.py | 23 |
| src/training/utils/regime_feature_utils.py | 23 |
| src/features_common/normalization.py | 23 |
| src/utils/confidence.py | 23 |
| src/utils/validated_step_factory.py | 23 |
| src/utils/logger.py | 23 |
| src/utils/ml_common/utils/feature_selection.py | 23 |
| src/utils/data/validation/validators.py | 23 |
| src/utils/hardware/memory_optimization.py | 23 |
| src/analyst/di_analyst.py | 23 |
| exchanges/shared/reliability/rate_limit_manager.py | 23 |
| src/trading/model_selection/model_selector_service.py | 22 |
| src/supervisor/dependency_container.py | 22 |
| src/supervisor/exchange_volume_adapter.py | 22 |
| src/feature_selection/specialized/adaptive_selector.py | 22 |
| src/training/common/component_result.py | 22 |
| src/training/steps/market_analysis/triple_barrier_validator.py | 22 |
| src/training/steps/market_analysis/coverage_constrained_clustering/utils.py | 22 |
| src/training/steps/market_analysis/optimal_regime_clustering_backup/performance_benchmark.py | 22 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/core/unified_architecture_search_engine.py | 22 |
| src/training/steps/market_analysis/regime_analysis/data_access.py | 22 |
| src/training/steps/market_analysis/components/hardware_setup.py | 22 |
| src/training/steps/market_analysis/components/sr_parameter_optimization.py | 22 |
| src/training/steps/market_analysis/tas_regime/optimization/enhanced_hardware_optimization.py | 22 |
| src/training/steps/market_analysis/tas_regime/core/search_space.py | 22 |
| src/training/steps/model_training/auto_step_trigger.py | 22 |
| src/features_common/optimization/cv_base.py | 22 |
| src/utils/error_handling_template.py | 22 |
| src/utils/step_validation_updater.py | 22 |
| src/utils/ml_common/models/hpo_enhancement_guide.py | 22 |
| src/utils/ml_common/optimization/shared_utils/advanced_metrics.py | 22 |
| src/utils/ml_common/training/training_utils.py | 22 |
| src/utils/hardware/enhanced_gpu_manager.py | 22 |
| src/feature_generation/auto_optimization_examples.py | 21 |
| src/feature_generation/core/auto_optimized_feature_generator.py | 21 |
| src/feature_selection/specialized/entropy_balancer.py | 21 |
| src/training/steps/data_collection/data_quality_components/quality_metrics_calculator.py | 21 |
| src/training/steps/market_analysis/labeling_components.py | 21 |
| src/training/steps/market_analysis/enhanced_market_analysis_with_triple_barrier.py | 21 |
| src/training/steps/market_analysis/clusters/engine.py | 21 |
| src/training/steps/market_analysis/clusters/clustering_orchestrator.py | 21 |
| src/features_common/mixins/monitoring_mixin.py | 21 |
| src/utils/step_validation_initializer.py | 21 |
| src/utils/ml_common/optimization/overfitting_prevention.py | 21 |
| src/utils/data/quality/comprehensive_quality_scorer.py | 21 |
| src/utils/matrix_operations/enhanced_operations.py | 21 |
| src/utils/common_ml/backtesting/monte_carlo_engine.py | 21 |
| exchanges/shared/wallet/balance_manager.py | 21 |
| src/feature_generation/test_default_auto_optimization.py | 20 |
| src/feature_generation/convenience/convenience_functions.py | 20 |
| src/feature_generation/core/optimization_strategies.py | 20 |
| src/feature_engineering_roadmap/feature_registry.py | 20 |
| src/tactician/position_closing.py | 20 |
| src/feature_selection/advanced/permutation_importance.py | 20 |
| src/training/steps/data_collection/data_quality_components/data_preprocessor.py | 20 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/enhanced_utility_integration.py | 20 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/core/regime_aware_training.py | 20 |
| src/training/steps/market_analysis/shared_utils/config.py | 20 |
| src/training/steps/market_analysis/shared_utils/characteristics.py | 20 |
| src/features_common/demo_extensive_logging.py | 20 |
| src/features_common/utils.py | 20 |
| src/utils/standardized_model_manager.py | 20 |
| src/utils/dependency_injection.py | 20 |
| src/utils/sr_clustering/sr_backtesting_engine.py | 20 |
| src/utils/ml_common/math_validation.py | 20 |
| src/utils/core/common.py | 20 |
| src/analyst/regime_runtime.py | 20 |
| src/monitoring/csv_exporter.py | 19 |
| src/models/stacker_lgbm_calibrated.py | 19 |
| src/supervisor/risk_allocator.py | 19 |
| src/tactician/ml_target_updater.py | 19 |
| src/tactician/leverage_sizer.py | 19 |
| src/training/steps/data_collection/unified_data_downloader.py | 19 |
| src/training/steps/market_analysis/components/sr_clustering.py | 19 |
| src/training/steps/market_analysis/nas_regime/core/adaptive_threshold_learning.py | 19 |
| src/training/steps/model_training/tactician_trainer.py | 19 |
| src/training/simplified_architecture/enhanced_interfaces.py | 19 |
| src/utils/ml_common/vectorbt_backtesting_engine.py | 19 |
| src/utils/ml_common/optimization/shared_utils/evaluation_metrics.py | 19 |
| src/utils/ml_common/reporting/validation_reporting_integration.py | 19 |
| src/deployment/rollout_plan.py | 19 |
| src/trading/monitoring/unified_trailing_manager.py | 18 |
| src/trading/cross_asset/trade_gate.py | 18 |
| src/feature_generation/categories/support_resistance.py | 18 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/metrics_reporting.py | 18 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/multi_timeframe_sync.py | 18 |
| src/training/steps/market_analysis/shared_utils/data_preprocessing.py | 18 |
| src/training/steps/market_analysis/tas_regime/utils/visualization.py | 18 |
| src/training/steps/market_analysis/nas_clustering/core/nas_regime_analyzer.py | 18 |
| src/training/steps/market_analysis/nas_modeling/core/nas_evaluator.py | 18 |
| src/training/steps/market_analysis/nas_modeling/core/neural_odes.py | 18 |
| src/training/steps/model_training/analyst_training_validation.py | 18 |
| src/training/steps/model_validation/tactician_validator.py | 18 |
| src/training/core/training_manager.py | 18 |
| src/core/examples/decorator_usage.py | 18 |
| src/features_common/error_handling.py | 18 |
| src/features_common/factories/scaler_factory.py | 18 |
| src/features_common/mixins/performance_mixin.py | 18 |
| src/utils/nonlinear_optimization_helpers.py | 18 |
| src/utils/cross_step_validation.py | 18 |
| src/utils/ml_common/vectorbt_performance_monitor.py | 18 |
| src/utils/ml_common/optimization/pure_tree_nas.py | 18 |
| src/utils/ml_common/evaluation/evaluation_utils.py | 18 |
| src/utils/ml_common/validation/enhanced_validation.py | 18 |
| research/feature_comparison/feature_versions.py | 18 |
| src/launcher/step_orchestrator_wrapper.py | 17 |
| src/trading/model_selection/trading_model_manager.py | 17 |
| src/supervisor/coordinator/system_coordinator.py | 17 |
| src/feature_generation/core/feature_cache.py | 17 |
| src/feature_generation/base_calculations/base_calculator.py | 17 |
| src/tactician/fully_migrated_tactician.py | 17 |
| src/training/steps/market_analysis/components/base_component.py | 17 |
| src/training/steps/market_analysis/nas_clustering/core/nas_search/evolutionary_search.py | 17 |
| src/training/utils/feature_selection/partial_information_decomposition.py | 17 |
| src/core/decorators/compose.py | 17 |
| src/utils/regime_ensemble_utils.py | 17 |
| src/utils/dependency_manager.py | 17 |
| src/utils/ml_common/optimization/tree_based_architecture_search.py | 17 |
| src/utils/ml_common/optimization/shared_utils/integration_verification.py | 17 |
| src/utils/ml_common/validation/underfitting_detection.py | 17 |
| src/analyst/order_book_analyzer.py | 17 |
| src/analyst/liquidation_risk_model.py | 17 |
| src/analyst/dynamic_regime_mapper.py | 17 |
| src/monitoring/enhanced_monitoring_launcher.py | 16 |
| src/launcher/configuration_manager.py | 16 |
| src/supervisor/coordinator/online_learning_manager.py | 16 |
| src/feature_generation/test_auto_optimization_integration.py | 16 |
| src/feature_generation/utils/optimized_feature_factory.py | 16 |
| src/training/steps/pre_training/column_naming.py | 16 |
| src/training/steps/market_analysis/optimal_regime_clustering_backup/optimized_clustering.py | 16 |
| src/training/steps/market_analysis/optimal_regime_clustering_backup/vectorized_operations.py | 16 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/automatic_training/regime_hpo_integration.py | 16 |
| src/training/steps/market_analysis/components/memory_manager.py | 16 |
| src/training/steps/market_analysis/optimized_multi_horizon_optimizer/enhanced_validation.py | 16 |
| src/training/steps/market_analysis/optimized_multi_horizon_optimizer/optimized_timeframe_optimizer.py | 16 |
| src/training/steps/market_analysis/clusters/features/selector.py | 16 |
| src/training/steps/model_training/patchtst_wrapper.py | 16 |
| src/utils/regime_data_access.py | 16 |
| src/utils/ml_common/optimization/hybrid_nas_system.py | 16 |
| src/utils/ml_common/explainability/shap_lime_integration.py | 16 |
| src/utils/hmm/__init__.py | 16 |
| src/utils/core/math_utilities.py | 16 |
| exchanges/binance/klines_adapter.py | 16 |
| exchanges/mexc/klines_adapter.py | 16 |
| exchanges/bingx/klines_adapter.py | 16 |
| exchanges/phemex/klines_adapter.py | 16 |
| exchanges/shared/wallet/balance_manager_old.py | 16 |
| exchanges/okx/klines_adapter.py | 16 |
| exchanges/gateio/klines_adapter.py | 16 |
| src/models/causal_dilated_tcn.py | 15 |
| src/supervisor/optimizer.py | 15 |
| src/feature_generation/categories/negative_learning.py | 15 |
| src/training/steps/market_analysis/optimal_regime_clustering_backup/optimized_integration.py | 15 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/unified_meta_learning.py | 15 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/unified_config_manager.py | 15 |
| src/training/steps/market_analysis/regime_analysis/service.py | 15 |
| src/training/steps/market_analysis/nas_modeling/core/meta_learning.py | 15 |
| src/training/steps/market_analysis/nas_modeling/core/neural_state_space_nas.py | 15 |
| src/core/dependency_injection.py | 15 |
| src/core/decorators/errors.py | 15 |
| src/utils/ml_common/models/model_cache.py | 15 |
| src/utils/ml_common/optimization/bayesian_entry_timing_optimizer.py | 15 |
| src/end_to_end_roadmap.py | 14 |
| src/trading/cross_asset/cross_asset_trading_manager.py | 14 |
| src/sentinel/sentinel.py | 14 |
| src/feature_generation/utils/optimized_cross_timeframe_analysis_methods.py | 14 |
| src/feature_generation/utils/statistical_calculations_optimizer.py | 14 |
| src/feature_selection/core/framework.py | 14 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/random_seed_manager.py | 14 |
| src/training/steps/market_analysis/optimization_cache.py | 14 |
| src/training/steps/market_analysis/tas_regime/production/monitoring.py | 14 |
| src/training/steps/market_analysis/regime_data_splitting/config_utils.py | 14 |
| src/training/steps/market_analysis/nas_regime/core/neural_architectures.py | 14 |
| src/config/sr_comprehensive_config_loader.py | 14 |
| src/utils/data_loader.py | 14 |
| src/utils/unified_utility_registry.py | 14 |
| src/utils/ml_common/optimization/hpo_diagnostics_and_fixes.py | 14 |
| src/utils/ml_common/data_processing/regime_processing.py | 14 |
| src/utils/matrix_operations/vectorized_correlations.py | 14 |
| src/nas_tas/monitoring/performance_monitor.py | 14 |
| src/nas_tas/evaluation/financial_metrics.py | 14 |
| src/nas_tas/config/validation_config.py | 14 |
| src/nas_tas/config/base_config.py | 14 |
| src/feature_generation/core/rolling_operations_mixin.py | 13 |
| src/feature_generation/utils/consolidated_rolling_optimizer.py | 13 |
| src/research/profit_labeling/test_enhanced_integration.py | 13 |
| src/tactician/enhanced_execution_manager.py | 13 |
| src/feature_selection/vectorbt/vectorbt_utils.py | 13 |
| src/training/steps/market_analysis/regime_aware_triple_barrier_optimizer.py | 13 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/core/financial_architecture_primitives.py | 13 |
| src/training/steps/market_analysis/tas_regime/verify_migration.py | 13 |
| src/training/steps/market_analysis/tas_regime/backtesting/walk_forward_analysis.py | 13 |
| src/training/steps/market_analysis/nas_clustering/core/nas_search/search_space.py | 13 |
| src/training/steps/model_training/xgboost_custom.py | 13 |
| src/training/utils/feature_selection/enhanced_partial_information_decomposition.py | 13 |
| src/core/enhanced_factories.py | 13 |
| src/features_common/demo_vectorbt_default.py | 13 |
| src/features_common/config/vectorbt_config.py | 13 |
| src/utils/memory_management/streaming_data_processor.py | 13 |
| src/utils/data/quality/data_qualification_config.py | 13 |
| src/utils/config/loaders.py | 13 |
| src/analyst/ml_dynamic_target_predictor.py | 13 |
| research/feature_comparison/robust_scaling.py | 13 |
| exchanges/shared/auth/auth_manager.py | 13 |
| src/trading/cross_asset/consolidated_reporting.py | 12 |
| src/feature_generation/core/feature_registry.py | 12 |
| src/research/crypto_analysis/data_downloader.py | 12 |
| src/feature_selection/sparse/sparse_feature_selector.py | 12 |
| src/feature_selection/optimizations/vectorized_operations.py | 12 |
| src/training/steps/data_collection/data_quality_components/data_integrity_checker.py | 12 |
| src/training/steps/data_collection/data_quality_components/error_handler.py | 12 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/ml_common_integration.py | 12 |
| src/training/steps/market_analysis/components/clustering_config.py | 12 |
| src/training/steps/market_analysis/components/sr_detection.py | 12 |
| src/training/steps/market_analysis/clusters/validation_framework.py | 12 |
| src/training/config/data_locator.py | 12 |
| src/core/di_integration.py | 12 |
| src/features_common/test_optimization_demo.py | 12 |
| src/features_common/test_backward_compatibility.py | 12 |
| src/utils/structured_logging.py | 12 |
| src/utils/enhanced_error_handler.py | 12 |
| src/utils/sr_clustering/backtesting_enhanced_clustering.py | 12 |
| src/utils/ml_common/vectorbt_memory_manager.py | 12 |
| src/utils/ml_common/validation/integrated_validation_system.py | 12 |
| src/utils/matrix_operations/__init__.py | 12 |
| research/cluster_analysis/clustering/optimal_cluster_selection.py | 12 |
| research/crypto_analysis/data_downloader.py | 12 |
| research/clusters/data_driven_clustering_framework.py | 12 |
| src/launcher/ARES_LAUNCHER_VERIFICATION.py | 11 |
| src/trading/config/trading_config.py | 11 |
| src/trading/config/execution_config.py | 11 |
| src/supervisor/loss_functions/pnl_calculator.py | 11 |
| src/feature_generation/utils/migration_helper.py | 11 |
| src/feature_generation/utils/step06_labeling_components/fractional_triple_barrier_labeling.py | 11 |
| src/feature_selection/caching/intelligent_feature_cache.py | 11 |
| src/training/steps/data_collection/data_downloader.py | 11 |
| src/training/steps/data_collection/data_preparation/run_step1.py | 11 |
| src/training/steps/market_analysis/standalone_optimizer.py | 11 |
| src/training/steps/market_analysis/coverage_constrained_clustering/run.py | 11 |
| src/training/steps/market_analysis/tas_regime/uncertainty/uncertainty_estimation.py | 11 |
| src/training/steps/market_analysis/tas_regime/core/tas_result.py | 11 |
| src/training/steps/market_analysis/tas_regime/core/tas_config.py | 11 |
| src/training/simplified_architecture/standard_interfaces.py | 11 |
| src/core/errors/mapping.py | 11 |
| src/features_common/config/optimization_config.py | 11 |
| src/utils/step_validation_wrapper.py | 11 |
| src/utils/state_manager.py | 11 |
| src/utils/ml_common/matrix_cross_validation.py | 11 |
| src/utils/ml_common/explainability/model_explainability.py | 11 |
| src/utils/ml_common/validation/universal_ml_validation.py | 11 |
| src/utils/ml_common/data_processing/data_cleaning_utils.py | 11 |
| research/feature_comparison/feature_comparison_utils.py | 11 |
| exchanges/shared/auth/api_key_manager.py | 11 |
| exchanges/shared/pricing/price_manager.py | 11 |
| src/monitoring/auto_monitoring_demo.py | 10 |
| src/research/crypto_analysis/run_optimized_analysis.py | 10 |
| src/feature_selection/vectorbt/vectorbt_config.py | 10 |
| src/training/steps/feature_engineering/filters/advanced_filters_15m.py | 10 |
| src/training/steps/market_analysis/nas_clustering/core/nas_regime_optimizer.py | 10 |
| src/training/steps/market_analysis/nas_regime/optimization/multi_objective_optimizer.py | 10 |
| src/core/di_launcher.py | 10 |
| src/features_common/backward_compatibility.py | 10 |
| src/config/pipeline_modes.py | 10 |
| src/config/computational_optimization_config.py | 10 |
| src/utils/serialization_utils.py | 10 |
| src/utils/ml_common/ensembles/stacking_ensemble_manager.py | 10 |
| src/database/firestore_manager.py | 10 |
| research/cluster_analysis/price_patterns/pattern_validation.py | 10 |
| research/crypto_analysis/run_optimized_analysis.py | 10 |
| exchanges/shared/auth/time_sync.py | 10 |
| exchanges/shared/orders/order_manager_old.py | 10 |
| src/supervisor/coordinator/recovery_manager.py | 9 |
| src/feature_generation/core/auto_optimization_config.py | 9 |
| src/research/crypto_analysis/run_analysis.py | 9 |
| src/tactician/sr_levels/sr_modules/sr_probability_calculator.py | 9 |
| src/training/steps/models_training/enhanced_entry_quality_scorer.py | 9 |
| src/training/steps/models_training/negative_learning_training_integration.py | 9 |
| src/training/steps/market_analysis/__init__.py | 9 |
| src/training/steps/market_analysis/coverage_constrained_clustering/component.py | 9 |
| src/training/steps/market_analysis/components/clustering_algorithms.py | 9 |
| src/training/steps/market_analysis/nas_modeling/core/nas_trainer.py | 9 |
| src/training/steps/market_analysis/regime_data_splitting/validation_utils.py | 9 |
| src/core/unified_config_service.py | 9 |
| src/features_common/logging_config.py | 9 |
| src/config/fractional_implementations_config.py | 9 |
| src/config/regime_feature_thresholds.py | 9 |
| src/utils/decorators/__init__.py | 9 |
| src/utils/ml_common/validation/__init__.py | 9 |
| src/utils/hmm/optimization.py | 9 |
| src/analyst/market_health_analyzer.py | 9 |
| research/crypto_analysis/run_analysis.py | 9 |
| GUI/verify_gui_workflow.py | 9 |
| exchanges/shared/interfaces_typed.py | 9 |
| exchanges/shared/pricing/ohlcv_manager.py | 9 |
| exchanges/shared/market/precision_helper.py | 9 |
| src/monitoring/integration_manager.py | 8 |
| src/launcher/gui_manager.py | 8 |
| src/models/tcn_regressor.py | 8 |
| src/feature_generation/__init__.py | 8 |
| src/feature_engineering_roadmap/interactions.py | 8 |
| src/feature_selection/advanced/adaptive_weighting.py | 8 |
| src/training/steps/data_collection/data_quality_components/config_manager.py | 8 |
| src/training/steps/backtesting/abc_testing/paper_trading_engine.py | 8 |
| src/training/steps/feature_engineering/register_features.py | 8 |
| src/training/steps/market_analysis/nas_clustering/core/nas_feature_extractor.py | 8 |
| src/training/steps/market_analysis/nas_regime/integration/nas_unified_integration.py | 8 |
| src/training/utils/embedding_postprocessing.py | 8 |
| src/core/errors/handlers/http.py | 8 |
| src/features_common/factories/optimizer_factory.py | 8 |
| src/config/regime_specific_optimization_config.py | 8 |
| src/utils/pipeline_results_manager.py | 8 |
| src/utils/compat.py | 8 |
| src/utils/artifact_manager.py | 8 |
| src/utils/ml_common/feature_selection_backwards_compat.py | 8 |
| src/utils/data/ares_launcher_data_loader.py | 8 |
| src/utils/data/quality/statistical_distribution_validation.py | 8 |
| src/utils/hardware/demo_implementation.py | 8 |
| src/nas_tas/config/search_config.py | 8 |
| research/cluster_analysis/price_patterns/__init__.py | 8 |
| research/cluster_analysis/clustering/__init__.py | 8 |
| research/clusters/dynamic_targets.py | 8 |
| examples/tactician_t1_t4_models_usage.py | 8 |
| data_quality/mapping/dependency_graph.py | 8 |
| exchanges/shared/orders/idempotency_manager.py | 8 |
| exchanges/shared/risk/risk_calculator.py | 8 |
| src/monitoring/correlation_manager.py | 7 |
| src/supervisor/pnl_loss_functions.py | 7 |
| src/supervisor/coordinator/health_monitor.py | 7 |
| src/training/steps/feature_engineering/feature_selector.py | 7 |
| src/training/steps/market_analysis/automatic_timeframe_optimizer.py | 7 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/unified_hardware_optimizer.py | 7 |
| src/training/steps/market_analysis/shared_utils/logging_utils.py | 7 |
| src/training/steps/market_analysis/shared_utils/calibration_registry.py | 7 |
| src/training/steps/market_analysis/tas_regime/integration/tas_unified_integration.py | 7 |
| src/training/steps/model_training/analyst_training_hardware.py | 7 |
| src/utils/purged_kfold.py | 7 |
| src/utils/artifact_pickup_utils.py | 7 |
| src/utils/ml_common/validation/thresholding.py | 7 |
| research/feature_comparison/method_settings.py | 7 |
| exchanges/shared/examples/high_level_usage.py | 7 |
| src/supervisor/global_portfolio_manager.py | 6 |
| src/supervisor/enhanced_model_monitor.py | 6 |
| src/supervisor/loss_functions/optimization_metrics.py | 6 |
| src/supervisor/coordinator/component_monitor.py | 6 |
| src/feature_engineering_roadmap/dynamic_feature_selector.py | 6 |
| src/training/steps/data_collection/data_preparation/step01_data_collection.py | 6 |
| src/training/steps/market_analysis/enhanced_multi_horizon_pipeline.py | 6 |
| src/training/steps/market_analysis/optimal_regime_clustering_backup/enhanced_clustering_integration.py | 6 |
| src/training/steps/market_analysis/tas_regime/regime_analysis/regime_reporting.py | 6 |
| src/training/steps/market_analysis/tas_regime/regime_analysis/regime_optimization.py | 6 |
| src/training/steps/market_analysis/tas_regime/utils/logging.py | 6 |
| src/core/domain.py | 6 |
| src/features_common/logging_enhancements.py | 6 |
| src/config/label_model_mapping.py | 6 |
| src/config/typed_config.py | 6 |
| src/utils/data_processing_utils.py | 6 |
| src/utils/caching.py | 6 |
| src/utils/ml_common/vectorbt_financial_metrics.py | 6 |
| src/utils/ml_common/config/universal_timeframe_config.py | 6 |
| src/utils/data/unified_data_utils.py | 6 |
| src/utils/data/real_data_loader.py | 6 |
| src/utils/data/__init__.py | 6 |
| src/database/influxdb_manager.py | 6 |
| research/clusters/__init__.py | 6 |
| examples/multi_stage_feature_selection_example.py | 6 |
| exchanges/shared/market/market_metadata.py | 6 |
| exchanges/base_exchange/exchange_interface.py | 6 |
| src/monitoring/ml_monitor.py | 5 |
| src/launcher/command_handlers.py | 5 |
| src/supervisor/loss_functions/loss_calculator.py | 5 |
| src/supervisor/loss_functions/performance_metrics.py | 5 |
| src/training/steps/data_collection/exchange_field_mappings.py | 5 |
| src/training/steps/market_analysis/gradient_flow_analysis.py | 5 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/regime_model_mapping/hybrid_integration.py | 5 |
| src/training/steps/market_analysis/tas_regime/regime_analysis/tree_regime_analyzer.py | 5 |
| src/training/steps/market_analysis/tas_regime/uncertainty/robustness_analysis.py | 5 |
| src/training/steps/market_analysis/nas_modeling/core/hardware_acceleration.py | 5 |
| src/training/steps/market_analysis/regime_model_mapping/nas_integration.py | 5 |
| src/training/steps/market_analysis/regime_model_mapping/tas_integration.py | 5 |
| src/training/steps/market_analysis/nas_regime/core/perfect_nas_config.py | 5 |
| src/training/steps/market_analysis/nas_regime/core/enhanced_perfect_nas_config.py | 5 |
| src/utils/math_validation.py | 5 |
| src/utils/sr_clustering/__init__.py | 5 |
| src/utils/ml_common/vectorized_backtesting.py | 5 |
| src/utils/ml_common/evaluation/enhanced_learning_curve_analysis.py | 5 |
| research/cluster_analysis/market_factor_analysis/__init__.py | 5 |
| torch_stub/__init__.py | 4 |
| src/trading/ensemble_disagreement_features.py | 4 |
| src/trading/utils/ohlcv.py | 4 |
| src/supervisor/loss_functions/base.py | 4 |
| src/supervisor/loss_functions/risk_metrics.py | 4 |
| src/training/steps/data_collection/data_quality_components/result_builder.py | 4 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/core/financial_loss_functions.py | 4 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/config/multi_timeframe_config.py | 4 |
| src/training/steps/market_analysis/tas_regime/evaluation/tree_evaluator.py | 4 |
| src/core/generic_base.py | 4 |
| src/config/environment.py | 4 |
| src/config/config_manager.py | 4 |
| src/config/validation.py | 4 |
| src/utils/monitoring_utils.py | 4 |
| src/utils/version_manager.py | 4 |
| src/utils/numba_timestamps.py | 4 |
| src/utils/hmm/core_manager.py | 4 |
| src/nas_tas/config/tprint_config.py | 4 |
| exchanges/shared/tests/verify_improvements.py | 4 |
| src/monitoring/performance_monitor.py | 3 |
| src/trading/nas_tas_trading_main.py | 3 |
| src/supervisor/monitoring.py | 3 |
| src/supervisor/coordinator/circuit_breaker.py | 3 |
| src/training/steps/backtesting/vectorbt_optimization_example.py | 3 |
| src/training/steps/market_analysis/regime_analysis/reporting.py | 3 |
| src/training/steps/market_analysis/tas_regime/evaluation/multi_objective_evaluation.py | 3 |
| src/training/steps/market_analysis/nas_clustering/core/nas_config.py | 3 |
| src/training/steps/market_analysis/nas_modeling/core/rl_nas.py | 3 |
| src/training/steps/market_analysis/clustering/config/clustering_config.py | 3 |
| src/training/steps/market_analysis/nas_regime/meta_learning/adaptive_regime_learner.py | 3 |
| src/features_common/factories/registry_factory.py | 3 |
| src/features_common/config/unified_config.py | 3 |
| src/config/sr_optimization_config.py | 3 |
| src/config/m1_gpu_config.py | 3 |
| src/config/analytical_process_config.py | 3 |
| src/config/computational_optimization.py | 3 |
| src/config/enhanced_reporting_config.py | 3 |
| src/utils/performance.py | 3 |
| src/utils/tracing.py | 3 |
| src/utils/random_seeding.py | 3 |
| src/utils/validation_decorators.py | 3 |
| src/utils/signal_handler.py | 3 |
| src/utils/tprint_integration.py | 3 |
| src/utils/prometheus_metrics.py | 3 |
| src/utils/ml_common/__init__.py | 3 |
| src/utils/ml_common/validation/temporal_cross_validation.py | 3 |
| src/utils/data/quality/gap_collection_hook.py | 3 |
| src/monitoring/advanced_tracer.py | 2 |
| src/monitoring/gui/launch_dashboard.py | 2 |
| src/tactician/enhanced_prediction_integrator.py | 2 |
| src/training/steps/pre_training/unified_data_driven_pipeline/core/simplified_config.py | 2 |
| src/training/steps/market_analysis/tas_regime/evaluation/regime_evaluation.py | 2 |
| src/core/service_registry.py | 2 |
| src/features_common/factories/unified_factory.py | 2 |
| src/features_common/registry/base_registry.py | 2 |
| src/config/multi_output_config.py | 2 |
| src/utils/observability.py | 2 |
| src/utils/regime_transition_handler.py | 2 |
| src/utils/ml_common/config/enhanced_ml_config.py | 2 |
| src/utils/core/data_types.py | 2 |
| src/utils/hardware/__init__.py | 2 |
| src/nas_tas/logging.py | 2 |
| src/monitoring/performance_dashboard.py | 1 |
| src/trading/examples/cross_asset_trading_demo.py | 1 |
| src/trading/examples/full_monitoring_demo.py | 1 |
| src/feature_generation/example_usage.py | 1 |
| src/feature_generation/categories/cross_timeframe.py | 1 |
| src/feature_generation/categories/regime_features.py | 1 |
| src/feature_generation/categories/enhanced_vectorbt_volatility.py | 1 |
| src/feature_generation/categories/interaction.py | 1 |
| src/feature_generation/tests/test_cleanup_validation.py | 1 |
| src/feature_generation/core/__init__.py | 1 |
| src/feature_generation/utils/optimized_feature_orchestrator.py | 1 |
| src/feature_generation/utils/sr_feature_extractor.py | 1 |
| src/feature_generation/utils/contrastive_learning_guide.py | 1 |
| src/feature_generation/utils/enhanced_optimization_system.py | 1 |
| src/feature_generation/utils/step06_comprehensive_implementation.py | 1 |
| src/feature_generation/utils/temporal_feature_integration.py | 1 |
| src/feature_generation/utils/optimized_cross_timeframe_analysis_advanced.py | 1 |
| src/feature_generation/utils/memory_optimizer.py | 1 |
| src/feature_generation/utils/cross_timeframe_interaction_features.py | 1 |
| src/feature_generation/utils/feature_generators_compatibility.py | 1 |
| src/feature_generation/utils/cross_timeframe_talib_integration.py | 1 |
| src/feature_generation/utils/__init__.py | 1 |
| src/feature_generation/utils/feature_generation_optimization.py | 1 |
| src/feature_generation/utils/statsmodels_integration.py | 1 |
| src/feature_generation/utils/optimization/lookback_optimizer.py | 1 |
| src/feature_generation/utils/step06_labeling_components/optimized_triple_barrier_labeling.py | 1 |
| src/feature_generation/utils/step06_labeling_components/regime_specific_triple_barrier_optimizer.py | 1 |
| src/feature_generation/utils/step06_labeling_components/regime_aware_triple_barrier_labeling.py | 1 |
| src/feature_generation/utils/step06_labeling_components/profit_based_feature_engineering.py | 1 |
| src/research/crypto_analysis/config.py | 1 |
| src/feature_engineering_roadmap/lookback_selection.py | 1 |
| src/feature_engineering_roadmap/ensemble_meta_features.py | 1 |
| src/tactician/position_monitor.py | 1 |
| src/tactician/sr_detection_optimization.py | 1 |
| src/tactician/sr_levels/enhanced_sr_detection.py | 1 |
| src/tactician/sr_levels/sr_breakout_predictor_enhanced.py | 1 |
| src/feature_selection/specialized/directional_selector.py | 1 |
| src/training/steps/data_collection/enhanced_klines_processing_pipeline.py | 1 |
| src/training/steps/data_collection/data_preparation/step01_5_data_converter.py | 1 |
| src/training/steps/data_collection/data_preparation/sr_strength_optimizer.py | 1 |
| src/training/steps/models_training/tactician_pre_ml_orchestration.py | 1 |
| src/training/steps/models_training/enhanced_tactician_pre_ml_orchestration.py | 1 |
| src/training/steps/models_training/tactician_models_training.py | 1 |
| src/training/steps/pre_training/unified_data_driven_pipeline/enhanced_components/vectorbt_enhancements.py | 1 |
| src/training/steps/pre_training/unified_data_driven_pipeline/steps/feature_generation_feature_selection_step.py | 1 |
| src/training/steps/market_analysis/regime_analysis_script.py | 1 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/shared_optimization.py | 1 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/evaluation/enhanced_regime_evaluator.py | 1 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/evaluation/robust_scoring_models.py | 1 |
| src/training/steps/market_analysis/hybrid_nas_tas_regime/core/nas_financial_features.py | 1 |
| src/training/steps/market_analysis/components/deprecated_nas_tas_clustering.py | 1 |
| src/training/steps/market_analysis/components/standardized_features.py | 1 |
| src/training/steps/market_analysis/tas_regime/regime_analysis/regime_qualification.py | 1 |
| src/training/steps/market_analysis/tas_regime/shared_utils/analysis_components.py | 1 |
| src/training/steps/market_analysis/tas_regime/shared_utils/search_strategies.py | 1 |
| src/training/steps/market_analysis/tas_regime/shared_utils/position_aware_trading.py | 1 |
| src/training/steps/market_analysis/tas_regime/data_pipeline/feature_engineering.py | 1 |
| src/training/steps/market_analysis/tas_regime/data_pipeline/data_preprocessing.py | 1 |
| src/training/steps/market_analysis/tas_regime/data_pipeline/regime_detection.py | 1 |
| src/training/steps/market_analysis/tas_regime/core/advanced_tas_search.py | 1 |
| src/training/steps/market_analysis/tas_regime/core/tas_regime_config.py | 1 |
| src/training/steps/market_analysis/tas_regime/core/tree_cvlSA_architecture.py | 1 |
| src/training/steps/market_analysis/regime_data_splitting/regime_data_splitting_main.py | 1 |
| src/training/steps/market_analysis/regime_data_splitting/regime_data_splitting_component.py | 1 |
| src/training/steps/market_analysis/regime_data_splitting/streamlined_regime_data_splitting.py | 1 |
| src/training/steps/market_analysis/nas_regime/evaluation/trading_viability_evaluator.py | 1 |
| src/training/steps/market_analysis/nas_regime/evaluation/economic_evaluator.py | 1 |
| src/training/steps/market_analysis/nas_regime/core/enhanced_data_operations.py | 1 |
| src/training/steps/model_training/tactician_lookback_optimization.py | 1 |
| src/training/utils/feature_calculators.py | 1 |
| src/training/simplified_architecture/modular_components.py | 1 |
| src/training/simplified_architecture/migrated_components/data_components.py | 1 |
| src/core/injectable_base.py | 1 |
| src/features_common/vectorbt/optimization_engine.py | 1 |
| src/features_common/vectorbt/gpu_accelerator.py | 1 |
| src/config/training_modes.py | 1 |
| src/config/multi_timeframe_hmm_ensemble_config.py | 1 |
| src/config/enhanced_matrix_config.py | 1 |
| src/config/__init__.py | 1 |
| src/config/trading.py | 1 |
| src/utils/intensity_scaler.py | 1 |
| src/utils/graceful_module_handler.py | 1 |
| src/utils/feature_engineering_validation.py | 1 |
| src/utils/enhanced_data_operations.py | 1 |
| src/utils/ml_common/vectorbt_memory_optimizer.py | 1 |
| src/utils/ml_common/optimization/unsupervised_tree_nas.py | 1 |
| src/utils/ml_common/optimization/trading_tree_architecture_search.py | 1 |
| src/utils/ml_common/optimization/regime_trading_tree_nas.py | 1 |
| src/utils/ml_common/data_processing/data_quality.py | 1 |
| src/utils/ml_common/data_processing/multi_timeframe_training.py | 1 |
| src/utils/ml_common/ensembles/__init__.py | 1 |
| src/utils/core/time_utilities.py | 1 |
| src/utils/common_ml/backtesting/turnover.py | 1 |
| src/nas_tas/results/comparison_utils.py | 1 |
| src/analyst/advanced_feature_engineering.py | 1 |
| src/analyst/feature_engineering_orchestrator.py | 1 |
| src/analyst/location_classifier_optimization.py | 1 |
| src/analyst/autoencoder_feature_generator.py | 1 |
| src/analyst/unified_regime_classifier.py | 1 |
| src/analyst/unified_regime_classifier_sr_optimized.py | 1 |
| src/analyst/analyst.py | 1 |
| src/analyst/predictive_ensembles/directional_specialist_model.py | 1 |
| src/analyst/predictive_ensembles/regime_ensembles/base_ensemble.py | 1 |
| live_trading/error_handler.py | 1 |
| research/crypto_analysis/config.py | 1 |
| research/clusters/feature_selection.py | 1 |
| examples/enhanced_label_definitions_demo.py | 1 |
| exchanges/shared/market/risk_tier_manager.py | 1 |
