#!/usr/bin/env python3

import sys
import os
from pathlib import Path

print("🚀 Starting simple test...")

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("📁 Project root added to path")

try:
    print("📦 Testing imports...")
    import numpy as np

    print("✅ numpy imported")

    import pandas as pd

    print("✅ pandas imported")

    print("📦 Testing src imports...")
    from src.config import CONFIG

    print("✅ CONFIG imported")

    from src.utils.logger import setup_logging, system_logger

    print("✅ logger imported")

    print("📦 Testing step1_5_data_converter...")
    from src.training.steps.step1_5_data_converter import UnifiedDataConverter

    print("✅ UnifiedDataConverter imported")

    print("🎉 All imports successful!")

except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback

    traceback.print_exc()
