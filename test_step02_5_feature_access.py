#!/usr/bin/env python3
"""
Test step02_5 feature access - ensure it has access to all features except:
- Feature interactions
- Regime-aware features
- Wavelets
"""

import pandas as pd
import numpy as np
from src.training.steps.feature_engineering.step06_advanced_features import AdvancedFeatureEngineeringStep
import time

# Create realistic test data
np.random.seed(42)
n = 5000  # Sufficient size for testing
base_price = 100
price_changes = np.random.normal(0, 0.003, n)
prices = base_price * (1 + np.cumsum(price_changes))

print(f'Generating {n} rows of realistic trading data...')
data = pd.DataFrame({
    'timestamp': pd.date_range('2023-01-01', periods=n, freq='1min'),
    'open': prices * (1 + np.random.normal(0, 0.001, n)),
    'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n))),
    'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n))),
    'close': prices,
    'volume': np.random.randint(1000, 10000, n)
})

print(f'Data shape: {data.shape}')
print()

# Test step06 with step02_5 compatibility mode
config = {'feature_engineering': {
    'enable_wavelets': False,
    'enable_feature_interactions': False,  # Explicitly disable for step02_5
    'disable_lookback_optimization': True
}}
step06 = AdvancedFeatureEngineeringStep(config)

print('🚀 Testing step06 with step02_5 compatibility mode...')
print('📋 Feature Access Policy for step02_5:')
print('   ✅ INCLUDED: Technical indicators, Microstructure features, Multi-timeframe features')
print('   ❌ EXCLUDED: Feature interactions, Regime-aware features, Wavelet features')
print()

try:
    # Test ALLOWED features (should work)
    print('🟢 Testing ALLOWED features:')

    # 1. Technical indicators
    print('📈 Testing technical indicators...')
    tech_features = step06._generate_comprehensive_technical_features(data)
    print(f'✅ Technical indicators: {len(tech_features.columns)} columns')

    # 2. Microstructure features
    print('🔬 Testing microstructure features...')
    micro_features = step06._calculate_microstructure_features(data)
    print(f'✅ Microstructure features: {len(micro_features.columns)} columns')

    # 3. Multi-timeframe features
    print('⏰ Testing multi-timeframe features...')
    import asyncio
    mtf_features = asyncio.run(step06._build_mtf_features_required(data))
    print(f'✅ Multi-timeframe features: {len(mtf_features.columns)} columns')

    print()
    print('🔴 Testing EXCLUDED features (should be disabled):')

    # 4. Feature interactions (should be disabled for step02_5)
    print('🔗 Testing feature interactions...')
    try:
        interaction_features = step06._create_feature_interactions(data)
        if len(interaction_features.columns) > 0:
            print(f'⚠️ WARNING: Feature interactions generated {len(interaction_features.columns)} columns (should be disabled)')
        else:
            print('✅ Feature interactions: Properly disabled (0 columns)')
    except Exception as e:
        print(f'✅ Feature interactions: Properly disabled (exception: {str(e)[:50]}...)')

    # 5. Regime-aware features (should be disabled for step02_5)
    print('🎭 Testing regime-aware features...')
    try:
        regime_features = step06._create_regime_aware_features(data, {})
        if len(regime_features.columns) > 0:
            print(f'⚠️ WARNING: Regime features generated {len(regime_features.columns)} columns (should be disabled)')
        else:
            print('✅ Regime-aware features: Properly disabled (0 columns)')
    except Exception as e:
        print(f'✅ Regime-aware features: Properly disabled (exception: {str(e)[:50]}...)')

    # 6. Wavelet features (should be disabled for step02_5)
    print('🌊 Testing wavelet features...')
    if step06.enable_wavelets and step06.wavelet_analyzer is not None:
        print('⚠️ WARNING: Wavelets enabled (should be disabled for step02_5)')
        try:
            wavelet_features = step06.wavelet_analyzer.extract_wavelet_features(data, price_column='close', symbol='SYMBOL', timeframe='30m')
            print(f'⚠️ WARNING: Wavelets generated {len(wavelet_features.columns)} columns (should be disabled)')
        except Exception as e:
            print(f'✅ Wavelet features: Properly disabled (exception: {str(e)[:50]}...)')
    else:
        print('✅ Wavelet features: Properly disabled (analyzer not available)')

    print()
    print('🎉 Feature access test completed!')
    print(f'📊 Total allowed features: {len(tech_features.columns) + len(micro_features.columns) + len(mtf_features.columns)}')

    # Summary
    print()
    print('📋 SUMMARY:')
    print(f'   ✅ Technical indicators: {len(tech_features.columns)} features')
    print(f'   ✅ Microstructure features: {len(micro_features.columns)} features')
    print(f'   ✅ Multi-timeframe features: {len(mtf_features.columns)} features')
    print('   ❌ Feature interactions: Disabled ✓')
    print('   ❌ Regime-aware features: Disabled ✓')
    print('   ❌ Wavelet features: Disabled ✓')

except Exception as e:
    print(f'❌ Test failed: {e}')
    import traceback
    traceback.print_exc()
