============================================================
📊 FEATURE GENERATION CODE QUALITY REPORT
============================================================
Generated: 2025-10-13 07:10:42

📈 SUMMARY METRICS
------------------------------
Total Files: 96
Total Lines: 84,421
Duplicate Methods: 12
Complex Methods: 0
Long Methods: 0
Large Files (>1000 lines): 31
Import Issues: 97
Documentation Coverage: 86.7%

🔄 DUPLICATE METHODS
------------------------------
core/feature_generator.py:def optimize_dataframe_processing\(: 1
core/feature_generator.py:def vectorized_rolling_operations\(: 1
core/optimization_mixin.py:def optimize_dataframe_processing\(: 1
core/vectorbt_optimization_mixin.py:def optimize_dataframe_processing\(: 1
utils/integration_updater.py:def optimize_dataframe_processing\(: 1
utils/integration_updater.py:def vectorized_rolling_operations\(: 1
utils/vectorization_optimizer.py:def optimize_dataframe_processing\(: 2
utils/vectorization_optimizer.py:def vectorized_rolling_operations\(: 2
validate_cleanup.py:def optimize_dataframe_processing\(: 1
validate_cleanup.py:def vectorized_rolling_operations\(: 1

📁 FILE COMPLEXITY
------------------------------
categories/regime_features.py: 4846 lines, 169 methods, 17 classes
categories/volume.py: 3662 lines, 123 methods, 32 classes
categories/trend.py: 2972 lines, 92 methods, 23 classes
utils/feature_generators.py: 2547 lines, 117 methods, 5 classes
utils/cross_timeframe_interaction_features.py: 2128 lines, 68 methods, 6 classes
categories/oscillator.py: 1976 lines, 50 methods, 13 classes
categories/microstructure_features.py: 1948 lines, 87 methods, 30 classes
categories/volatility.py: 1746 lines, 52 methods, 14 classes
categories/returns.py: 1684 lines, 46 methods, 15 classes
categories/momentum.py: 1540 lines, 52 methods, 19 classes

⚠️ IMPORT ISSUES
------------------------------
categories/advanced_statistical.py:79: Multiple imports on one line
categories/advanced_statistical.py:92: Multiple imports on one line
categories/volume.py:27: Multiple imports on one line
categories/volume.py:89: Multiple imports on one line
categories/cross_timeframe.py:25: Multiple imports on one line
categories/cross_timeframe.py:56: Multiple imports on one line
categories/cross_timeframe.py:65: Multiple imports on one line
categories/oscillator.py:14: Multiple imports on one line
categories/oscillator.py:69: Multiple imports on one line
categories/oscillator.py:115: Multiple imports on one line
... and 87 more issues

🎯 QUALITY SCORE
------------------------------
Overall Score: 60.0/100
Status: 🔴 NEEDS IMPROVEMENT

💡 RECOMMENDATIONS
------------------------------
• Remove remaining duplicate methods
• Consider splitting large files
• Fix import issues and avoid wildcard imports

============================================================