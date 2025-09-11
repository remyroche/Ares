#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Minimal debug script to isolate the hanging import
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

tprint("Step 1: Starting minimal debug...")

try:
    tprint("Step 2: Importing simple_logger...")
    from simple_logger import system_logger
    tprint("Step 3: simple_logger imported OK")
except Exception as e:
    tprint(f"ERROR in simple_logger: {e}")
    sys.exit(1)

try:
    tprint("Step 4: Importing src.core.decorators...")
    from src.core.decorators import handles_errors
    tprint("Step 5: decorators imported OK")
except Exception as e:
    tprint(f"ERROR in decorators: {e}")
    sys.exit(1)

tprint("Step 6: About to import main_training_pipeline...")
try:
    from src.training.steps.main_training_pipeline import MainTrainingPipeline
    tprint("Step 7: main_training_pipeline imported OK")
except Exception as e:
    tprint(f"ERROR in main_training_pipeline: {e}")
    sys.exit(1)

tprint("Step 8: All imports successful!")
