#!/usr/bin/env python3
"""
Test script to verify HMM mode auto-detection mapping.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.training.steps.main_training_pipeline import ExecutionMode
from src.utils.hmm_composite_manager import HMMCompositeManager

def test_mode_mapping():
    """Test the auto-detection of HMM modes based on launcher modes."""

    # Create mock configs with different modes
    configs = {
        'FULL': type('MockConfig', (), {'mode': ExecutionMode.FULL})(),
        'LIGHT': type('MockConfig', (), {'mode': ExecutionMode.LIGHT})(),
        'BLANK': type('MockConfig', (), {'mode': ExecutionMode.BLANK})(),
        'NONE': None  # Test default case
    }

    # Initialize HMM manager
    hmm_manager = HMMCompositeManager()

    print("🔧 Testing HMM Mode Auto-Detection Mapping")
    print("=" * 50)

    expected_mappings = {
        'FULL': 'FULL',
        'LIGHT': 'BLANK',
        'BLANK': 'LIGHT',
        'NONE': 'BLANK'
    }

    for config_name, config in configs.items():
        detected_mode = hmm_manager._auto_detect_hmm_mode(config)
        expected_mode = expected_mappings[config_name]

        status = "✅" if detected_mode == expected_mode else "❌"
        print(f"{status} {config_name} mode → HMM {detected_mode} mode (expected: {expected_mode})")

        # Show parameter ranges for the detected mode
        param_ranges = hmm_manager._get_hmm_parameter_ranges()
        mode_params = param_ranges[detected_mode]
        print(f"   📊 n_components: {mode_params['n_components_min']}-{mode_params['n_components_max']}")
        print(f"   📊 n_iter: {mode_params['n_iter_min']}-{mode_params['n_iter_max']}")
        print(f"   📊 tol: {mode_params['tol_min']}-{mode_params['tol_max']}")
        print(f"   📝 {mode_params['description']}")
        print()

if __name__ == "__main__":
    test_mode_mapping()
