#!/usr/bin/env python3
"""
Minimal debug script to isolate the hanging import
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("Step 1: Starting minimal debug...")

try:
    print("Step 2: Importing simple_logger...")
    from simple_logger import system_logger
    print("Step 3: simple_logger imported OK")
except Exception as e:
    print(f"ERROR in simple_logger: {e}")
    sys.exit(1)

try:
    print("Step 4: Importing src.core.decorators...")
    from src.core.decorators import handles_errors
    print("Step 5: decorators imported OK")
except Exception as e:
    print(f"ERROR in decorators: {e}")
    sys.exit(1)

print("Step 6: About to import main_training_pipeline...")
try:
    from src.training.steps.main_training_pipeline import MainTrainingPipeline
    print("Step 7: main_training_pipeline imported OK")
except Exception as e:
    print(f"ERROR in main_training_pipeline: {e}")
    sys.exit(1)

print("Step 8: All imports successful!")
