#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Debug script to isolate where the launcher hangs
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

tprint("1. Starting debug script...")
tprint(f"1.1. Python path: {sys.path[:3]}")

# Import simple logger first
tprint("1.2. About to import simple_logger...")
from simple_logger import system_logger
tprint("2. Simple logger imported successfully")

# Try importing the problematic modules one by one
try:
    tprint("3. About to import decorators...")
    tprint("3.1. Importing src.core.decorators...")
    from src.core.decorators import handles_errors, traced, log_execution_time
    tprint("4. Decorators imported successfully")
except Exception as e:
    tprint(f"4. ERROR importing decorators: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    tprint("5. About to import main training pipeline components...")
    tprint("5.1. Importing src.training.steps.main_training_pipeline...")
    from src.training.steps.main_training_pipeline import (
        MainTrainingPipeline, MainPipelineConfig, MainPipelineResult,
        PipelineStage, ExecutionMode, get_full_pipeline_config,
        get_light_pipeline_config, get_blank_pipeline_config, SubPipelineStatus
    )
    tprint("6. Main training pipeline components imported successfully")
except Exception as e:
    tprint(f"6. ERROR importing main training pipeline components: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    tprint("7. About to create MainTrainingPipeline instance...")
    tprint("7.1. Calling MainTrainingPipeline()...")
    pipeline = MainTrainingPipeline()
    tprint("8. MainTrainingPipeline created successfully")
except Exception as e:
    tprint(f"8. ERROR creating MainTrainingPipeline: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

tprint("9. All imports and initialization completed successfully!")
