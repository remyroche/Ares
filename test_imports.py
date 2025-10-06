#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, '/workspace')

print("Testing imports...")

try:
    from src.utils.logger import system_logger
    print("✅ Logger imported successfully")
except ImportError as e:
    print(f"❌ Logger import failed: {e}")
    import traceback
    traceback.print_exc()

try:
    from src.training.steps.models_training.tactician_pre_ml_orchestration import TacticianPreMLOrchestrator
    print("✅ TacticianPreMLOrchestrator imported successfully")
except ImportError as e:
    print(f"❌ TacticianPreMLOrchestrator import failed: {e}")
    import traceback
    traceback.print_exc()

print("Import testing completed.")