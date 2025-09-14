#!/usr/bin/env python3
"""
Test script for new matrix operations: safe_divide and ewm_mean
"""

import numpy as np
import pandas as pd
import sys
import traceback

# Add the src directory to the path
sys.path.insert(0, '/Users/remyroche/Documents/Ares/src')

def test_new_operations():
    """Test the new safe_divide and ewm_mean operations."""
    print('🚀 Testing new matrix operations:')
    print('=' * 50)

    try:
        from src.utils.ml_common.matrix_operations import get_unified_matrix_operations
        print('✅ Successfully imported matrix operations')

        # Initialize matrix operations
        ops = get_unified_matrix_operations()
        print('✅ Matrix operations initialized')

        # Create sample data
        np.random.seed(42)
        data = np.random.randn(100, 5)
        print(f'📊 Created sample data: {data.shape}')

        # Test safe_divide operation
        print('\n🧮 Testing safe_divide operation...')
        try:
            numerator = np.array([1, 2, 3, 4, 5])
            denominator = np.array([1, 2, 0, 4, 0])  # Contains zeros

            print(f'Numerator: {numerator}')
            print(f'Denominator: {denominator}')

            result = ops.batch_process(data, 'safe_divide',
                                     numerator=numerator,
                                     denominator=denominator,
                                     default_value=999)

            print(f'Safe divide result type: {type(result)}')
            if hasattr(result, 'shape'):
                print(f'Safe divide result shape: {result.shape}')
                print(f'Safe divide with zeros: {result[0]}')
            else:
                print(f'Safe divide result: {result}')

        except Exception as e:
            print(f'❌ Safe divide test failed: {e}')
            traceback.print_exc()

        # Test ewm_mean operation
        print('\n📊 Testing ewm_mean operation...')
        try:
            ewm_result = ops.batch_process(data, 'ewm_mean', span=10, adjust=True)
            print(f'EWM mean result type: {type(ewm_result)}')
            if hasattr(ewm_result, 'shape'):
                print(f'EWM mean result shape: {ewm_result.shape}')
                print(f'EWM mean sample values: {ewm_result[0, :3]}')
            else:
                print(f'EWM mean result: {ewm_result}')

        except Exception as e:
            print(f'❌ EWM mean test failed: {e}')
            traceback.print_exc()

        print('\n✅ New operations test completed!')

    except Exception as e:
        print(f'❌ Test failed with error: {e}')
        traceback.print_exc()

if __name__ == "__main__":
    test_new_operations()
