#!/usr/bin/env python3
"""
Test step06 compatibility with step02_5 using 20,000 rows of realistic trading data
"""

import pandas as pd
import numpy as np
from src.training.steps.feature_engineering.step06_advanced_features import AdvancedFeatureEngineeringStep
import time

# Create very large test data - realistic for production ML training
np.random.seed(42)
n = 20000  # Even larger dataset - ~2 weeks of 1-minute trading data
base_price = 100
price_changes = np.random.normal(0, 0.003, n)  # Small price changes
prices = base_price * (1 + np.cumsum(price_changes))

print(f'Generating {n} rows of realistic trading data...')
start_time = time.time()

data = pd.DataFrame({
    'timestamp': pd.date_range('2023-01-01', periods=n, freq='1min'),
    'open': prices * (1 + np.random.normal(0, 0.001, n)),
    'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n))),
    'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n))),
    'close': prices,
    'volume': np.random.randint(1000, 10000, n)
})

gen_time = time.time() - start_time
print(f'✅ Data generation completed in {gen_time:.2f}s')
print(f'Data shape: {data.shape}')
print(f'Time range: {data.timestamp.min()} to {data.timestamp.max()}')
print(f'Price range: ${data.close.min():.2f} to ${data.close.max():.2f}')
print(f'Average volume: {data.volume.mean():.0f}')
print()

# Test step06 with step02_5 compatibility mode
config = {'feature_engineering': {'enable_wavelets': False, 'disable_lookback_optimization': True}}
step06 = AdvancedFeatureEngineeringStep(config)

print('🚀 Testing step06 with step02_5 compatibility mode (20,000 rows)...')

try:
    # Test the methods that step02_5 tries to call
    test_start = time.time()

    print('📈 Testing _generate_comprehensive_technical_features...')
    tech_features = step06._generate_comprehensive_technical_features(data)
    tech_time = time.time() - test_start
    print(f'✅ Technical features: {len(tech_features.columns)} columns in {tech_time:.2f}s')

    print('🔬 Testing _calculate_microstructure_features...')
    micro_start = time.time()
    micro_features = step06._calculate_microstructure_features(data)
    micro_time = time.time() - micro_start
    print(f'✅ Microstructure features: {len(micro_features.columns)} columns in {micro_time:.2f}s')

    print('🔗 Testing _create_feature_interactions...')
    interaction_start = time.time()
    interaction_features = step06._create_feature_interactions(data)
    interaction_time = time.time() - interaction_start
    print(f'✅ Feature interactions: {len(interaction_features.columns)} columns in {interaction_time:.2f}s')

    print('🎭 Testing _create_regime_aware_features...')
    regime_start = time.time()
    regime_features = step06._create_regime_aware_features(data, {})
    regime_time = time.time() - regime_start
    print(f'✅ Regime-aware features: {len(regime_features.columns)} columns in {regime_time:.2f}s')

    print('🔧 Testing regime_engine attribute...')
    print(f'✅ Regime engine: {type(step06.regime_engine)}')

    total_time = time.time() - start_time
    print()
    print('🎉 All step02_5 compatibility tests passed!')
    print(f'📊 Total execution time: {total_time:.2f}s')
    print(f'📈 Total features generated: {len(tech_features.columns) + len(micro_features.columns) + len(interaction_features.columns) + len(regime_features.columns)}')

    # Check for any NaN issues in the results
    print()
    print('🔍 Feature quality check:')
    all_features = [tech_features, micro_features, interaction_features, regime_features]
    total_nan_pct = 0

    for i, features in enumerate(all_features):
        nan_cols = features.isnull().sum()
        if len(nan_cols) > 0:
            max_nan = nan_cols.max()
            max_nan_pct = (max_nan / len(features)) * 100
            total_nan_pct = max(total_nan_pct, max_nan_pct)
            print(f'  Feature group {i+1}: Max NaN = {max_nan_pct:.2f}%')

    if total_nan_pct < 1.0:
        print(f'✅ Excellent: All features have < 1% NaN values (max: {total_nan_pct:.2f}%)')
    elif total_nan_pct < 5.0:
        print(f'✅ Good: All features have < 5% NaN values (max: {total_nan_pct:.2f}%)')
    else:
        print(f'⚠️ Warning: Some features have high NaN values (max: {total_nan_pct:.2f}%)')
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
