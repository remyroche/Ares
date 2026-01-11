
import sys
import os
import logging

# Configure logging to capture any import warnings/errors
logging.basicConfig(level=logging.DEBUG)

sys.path.insert(0, os.getcwd())

print("Attempting to import src.training.steps.labeling...")
try:
    import src.training.steps.labeling
    print("✅ Successfully imported src.training.steps.labeling")
except Exception as e:
    print(f"❌ Failed to import src.training.steps.labeling: {e}")
    import traceback
    traceback.print_exc()

print("\nChecking Step Registry...")
try:
    from src.training.steps.base_step import step_registry
    steps = step_registry.list_steps()
    print(f"Found {len(steps)} steps in registry.")
    if "meta_labeling_hpo_sample_weighted" in steps:
        print("✅ 'meta_labeling_hpo_sample_weighted' FOUND in registry.")
    else:
        print("❌ 'meta_labeling_hpo_sample_weighted' NOT FOUND in registry.")
        print(f"Available steps: {steps}")
except Exception as e:
    print(f"❌ Failed to check registry: {e}")
