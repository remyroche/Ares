#!/usr/bin/env python3
"""
Debug script to isolate where the launcher hangs
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("1. Starting debug script...")
print(f"1.1. Python path: {sys.path[:3]}")

# Import simple logger first
print("1.2. About to import simple_logger...")
from simple_logger import system_logger
print("2. Simple logger imported successfully")

# Try importing the problematic modules one by one
try:
    print("3. About to import decorators...")
    print("3.1. Importing src.core.decorators...")
    from src.core.decorators import handles_errors, traced, log_execution_time
    print("4. Decorators imported successfully")
except Exception as e:
    print(f"4. ERROR importing decorators: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("5. About to import main training pipeline components...")
    print("5.1. Importing src.training.steps.main_training_pipeline...")
    from src.training.steps.main_training_pipeline import (
        MainTrainingPipeline, MainPipelineConfig, MainPipelineResult,
        PipelineStage, ExecutionMode, get_full_pipeline_config,
        get_light_pipeline_config, get_blank_pipeline_config, SubPipelineStatus
    )
    print("6. Main training pipeline components imported successfully")
except Exception as e:
    print(f"6. ERROR importing main training pipeline components: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("7. About to create MainTrainingPipeline instance...")
    print("7.1. Calling MainTrainingPipeline()...")
    pipeline = MainTrainingPipeline()
    print("8. MainTrainingPipeline created successfully")
except Exception as e:
    print(f"8. ERROR creating MainTrainingPipeline: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("9. All imports and initialization completed successfully!")
