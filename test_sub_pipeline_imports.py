#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Test script to isolate which sub-pipeline import is causing the hang
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

tprint("1. Starting sub-pipeline import test...")

try:
    tprint("2. About to import simple_logger...")
    from simple_logger import system_logger
    tprint("3. Simple logger imported successfully")
except Exception as e:
    tprint(f"3. ERROR importing simple_logger: {e}")
    sys.exit(1)

try:
    tprint("4. About to import decorators...")
    from src.core.decorators import handles_errors, traced, log_execution_time
    tprint("5. Decorators imported successfully")
except Exception as e:
    tprint(f"5. ERROR importing decorators: {e}")
    sys.exit(1)

# Test data collection sub-pipeline import
try:
    tprint("6. About to import data_collection sub_pipeline...")
    from src.training.steps.data_collection.sub_pipeline import (
        DataCollectionSubPipeline, SubPipelineConfig as DataCollectionConfig,
        SubPipelineResult as DataCollectionResult, ExecutionMode, SubPipelineStatus
    )
    tprint("7. Data collection sub-pipeline imported successfully")
except Exception as e:
    tprint(f"7. ERROR importing data_collection sub_pipeline: {e}")
    import traceback
    traceback.print_exc()

tprint("8. Test completed!")

