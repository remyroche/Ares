#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Minimal import test to find the exact hanging point
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

tprint("1. Starting minimal import test...")

# Test basic imports first
tprint("2. Testing basic Python imports...")
import os
import sys
import logging
tprint("3. Basic imports successful")

# Test simple_logger
tprint("4. Testing simple_logger...")
from simple_logger import system_logger
tprint("5. Simple logger successful")

# Test decorators
tprint("6. Testing decorators...")
from src.core.decorators import handles_errors
tprint("7. Decorators successful")

# Test if the issue is in the src.training.steps package
tprint("8. Testing src.training.steps package...")
import src.training.steps
tprint("9. src.training.steps package successful")

# Test if the issue is in the data_collection module
tprint("10. Testing data_collection module...")
import src.training.steps.data_collection
tprint("11. data_collection module successful")

# Test if the issue is in the sub_pipeline module
tprint("12. Testing sub_pipeline module...")
import src.training.steps.data_collection.sub_pipeline
tprint("13. sub_pipeline module successful")

tprint("14. All minimal imports completed successfully!")

