#!/usr/bin/env python3
"""
Run Sticky Finite HMM with fresh module imports to avoid cached issues.
This script ensures the fixed regime generators are loaded properly.
"""

import sys
import os

# Clear all cached modules related to feature generation
modules_to_clear = []
for module_name in sys.modules:
    if any(x in module_name for x in [
        'src.feature_generation.categories.advanced_regime_features',
        'src.feature_generation.categories',
        'src.feature_generation',
        'src.training.steps.market_analysis.sticky_finite_hmm_clustering'
    ]):
        modules_to_clear.append(module_name)

for module_name in modules_to_clear:
    del sys.modules[module_name]

print(f"Cleared {len(modules_to_clear)} cached modules")

# Now run the main pipeline
exec(open('run_sticky_finite_hmm_complete.py').read())
