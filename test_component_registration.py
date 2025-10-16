#!/usr/bin/env python3
"""
Test script to verify component registration is working.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.training.steps.pre_training.components import ComponentFactory
    from src.training.steps.pre_training.components.component_registry import ComponentRegistry
    
    print("🔍 Testing component registration...")
    
    # Get available components
    available_components = ComponentFactory.get_available_components()
    print(f"✅ Available components: {available_components}")
    
    # Get registration status
    registration_status = ComponentRegistry.get_registration_status()
    print(f"📊 Registration status:")
    for component, status in registration_status.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {component}: {'Available' if status else 'Not Available'}")
    
    # Check if all expected components are registered
    expected_components = [
        'analyst_profit_labeler',
        'tactician_entry_labeler',
        'feature_generation_data_validation_step',
        'feature_generation_feature_generation_step',
        'feature_generation_feature_selection_step',
        'feature_generation_final_validation_step',
        'feature_generation_interaction_generation_step',
        'feature_generation_labeling_integration_step',
        'feature_generation_period_lookback_optimization',
        'feature_generation_vectorization_step'
    ]
    
    missing_components = []
    for component in expected_components:
        if component not in available_components:
            missing_components.append(component)
    
    if missing_components:
        print(f"❌ Missing components: {missing_components}")
        sys.exit(1)
    else:
        print("✅ All expected components are registered!")
        sys.exit(0)
        
except Exception as e:
    print(f"❌ Error testing component registration: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
