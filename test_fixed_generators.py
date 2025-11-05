#!/usr/bin/env python3
"""Test script to verify the fixed regime generators."""

import sys
import importlib

# Force reload of modules
if 'src.feature_generation.categories.advanced_regime_features' in sys.modules:
    del sys.modules['src.feature_generation.categories.advanced_regime_features']
if 'src.feature_generation.categories' in sys.modules:
    del sys.modules['src.feature_generation.categories']
if 'src.feature_generation' in sys.modules:
    del sys.modules['src.feature_generation']

# Now import fresh
sys.path.insert(0, 'src')

from src.feature_generation.categories.advanced_regime_features import (
    RegimeEntropyGenerator,
    RegimeComplexityGenerator,
    RegimeFractalDimensionGenerator,
    RegimeHurstExponentGenerator,
    RegimeMemoryStrengthGenerator
)
from src.utils.data_loader import DataLoader
import pandas as pd
import numpy as np

def test_generators():
    """Test all fixed generators."""
    print("Testing fixed regime generators...")
    
    # Load test data
    loader = DataLoader()
    data = loader.load_ethusdt_1h_data()
    if data is None or len(data) < 100:
        print('❌ No test data available')
        return False
    
    # Use smaller sample for testing
    test_data = data.head(200).copy()
    print(f'Testing with {len(test_data)} samples')
    
    # Test each generator
    generators = [
        ('RegimeEntropyGenerator', RegimeEntropyGenerator()),
        ('RegimeComplexityGenerator', RegimeComplexityGenerator()),
        ('RegimeFractalDimensionGenerator', RegimeFractalDimensionGenerator()),
        ('RegimeHurstExponentGenerator', RegimeHurstExponentGenerator()),
        ('RegimeMemoryStrengthGenerator', RegimeMemoryStrengthGenerator())
    ]
    
    all_passed = True
    total_features = 0
    
    for name, generator in generators:
        try:
            # Check if generate_features method exists
            if not hasattr(generator, 'generate_features'):
                print(f'❌ {name}: Missing generate_features method')
                all_passed = False
                continue
                
            features = generator.generate_features(test_data)
            if features and len(features) > 0:
                print(f'✅ {name}: Generated {len(features)} features')
                total_features += len(features)
                for feat_name, feat_array in features.items():
                    if isinstance(feat_array, np.ndarray) and len(feat_array) > 0:
                        valid_count = np.sum(~np.isnan(feat_array))
                        print(f'   - {feat_name}: shape {feat_array.shape}, valid: {valid_count}')
            else:
                print(f'⚠️ {name}: No features generated')
                all_passed = False
        except Exception as e:
            print(f'❌ {name}: Failed with {e}')
            all_passed = False
    
    if all_passed:
        print(f'\n✅ All generators working! Total: {total_features} features')
    else:
        print('\n⚠️ Some generators have issues')
    
    return all_passed

if __name__ == "__main__":
    success = test_generators()
    sys.exit(0 if success else 1)
