#!/usr/bin/env python3
"""
Minimal import test to find the exact hanging point
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("1. Starting minimal import test...")

# Test basic imports first
print("2. Testing basic Python imports...")
import os
import sys
import logging
print("3. Basic imports successful")

# Test simple_logger
print("4. Testing simple_logger...")
from simple_logger import system_logger
print("5. Simple logger successful")

# Test decorators
print("6. Testing decorators...")
from src.core.decorators import handles_errors
print("7. Decorators successful")

# Test if the issue is in the src.training.steps package
print("8. Testing src.training.steps package...")
import src.training.steps
print("9. src.training.steps package successful")

# Test if the issue is in the data_collection module
print("10. Testing data_collection module...")
import src.training.steps.data_collection
print("11. data_collection module successful")

# Test if the issue is in the sub_pipeline module
print("12. Testing sub_pipeline module...")
import src.training.steps.data_collection.sub_pipeline
print("13. sub_pipeline module successful")

print("14. All minimal imports completed successfully!")

