
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

print("Attempting to import src.training.steps.labeling.label_based_layer_1...")

try:
    import src.training.steps.labeling.label_based_layer_1
    print("Import successful!")
except Exception as e:
    print(f"Import failed with error: {e}")
    import traceback
    traceback.print_exc()

print("Done.")
