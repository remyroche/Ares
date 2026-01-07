
import sys
import os
sys.path.insert(0, os.getcwd())

print("Attempting to import MetaLabelingHPOSampleWeightedStep...")
try:
    from src.training.steps.labeling.meta_labeling_hpo_sample_weighted import MetaLabelingHPOSampleWeightedStep
    print("SUCCESS: Import successful")
except Exception as e:
    print(f"FAILURE: {e}")
    import traceback
    traceback.print_exc()

print("\nAttempting to import LabelBasedLayer2...")
try:
    from src.training.steps.labeling.label_based_layer_2 import LabelBasedLayer2
    print("SUCCESS: LabelBasedLayer2 imported")
except Exception as e:
    print(f"FAILURE: {e}")
    import traceback
    traceback.print_exc()
