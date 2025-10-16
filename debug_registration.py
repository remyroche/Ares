#!/usr/bin/env python3
"""
Debug script to check component registration.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    print("🔍 Testing component registration step by step...")
    
    # Test 1: Import ComponentFactory
    print("1. Importing ComponentFactory...")
    from src.training.steps.pre_training.components import ComponentFactory
    print("✅ ComponentFactory imported successfully")
    
    # Test 2: Check if components are registered
    print("2. Checking available components...")
    available_components = ComponentFactory.get_available_components()
    print(f"Available components: {available_components}")
    
    # Test 3: Try to import the component registry directly
    print("3. Importing component registry...")
    from src.training.steps.pre_training.components.component_registry import ComponentRegistry
    print("✅ ComponentRegistry imported successfully")
    
    # Test 4: Check registration status
    print("4. Checking registration status...")
    registration_status = ComponentRegistry.get_registration_status()
    print(f"Registration status: {registration_status}")
    
    # Test 5: Try to register components manually
    print("5. Trying to register components manually...")
    ComponentRegistry.register_all_components()
    
    # Test 6: Check components again
    print("6. Checking components after manual registration...")
    available_components = ComponentFactory.get_available_components()
    print(f"Available components after registration: {available_components}")
    
    sys.exit(0)
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
