import sys
import os
sys.path.insert(0, os.getcwd())

try:
    print("Attempting to import train_specialists_with_gmm_step...")
    from src.training.steps.labeling import train_specialists_with_gmm_step
    print("Import successful!")
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Cheking CausalDiscoveryLight ---")
try:
    from src.training.steps.labeling import causal_discovery_light
    print("CausalDiscoveryLight imported successfully!")
except Exception as e:
    print(f"CausalDiscoveryLight import failed: {e}")
    import traceback
    traceback.print_exc()
