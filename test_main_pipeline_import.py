#!/usr/bin/env python3
"""
Test script to isolate the main_training_pipeline import issue
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("1. Starting main pipeline import test...")

try:
    print("2. About to import simple_logger...")
    from simple_logger import system_logger
    print("3. Simple logger imported successfully")
except Exception as e:
    print(f"3. ERROR importing simple_logger: {e}")
    sys.exit(1)

try:
    print("4. About to import decorators...")
    from src.core.decorators import handles_errors, traced, log_execution_time
    print("5. Decorators imported successfully")
except Exception as e:
    print(f"5. ERROR importing decorators: {e}")
    sys.exit(1)

try:
    print("6. About to import main_training_pipeline...")
    from src.training.steps.main_training_pipeline import (
        MainTrainingPipeline, MainPipelineConfig, MainPipelineResult,
        PipelineStage, ExecutionMode, get_full_pipeline_config,
        get_light_pipeline_config, get_blank_pipeline_config, SubPipelineStatus
    )
    print("7. Main training pipeline imported successfully")
except Exception as e:
    print(f"7. ERROR importing main_training_pipeline: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("8. All imports completed successfully!")

