import sys
import os

# Add src to path
sys.path.append(os.getcwd())

print("Testing Layer 3 imports...")
failed = False

try:
    from src.training.steps.labeling.layer3 import core
    print("✅ layer3.core imported")
except Exception as e:
    print(f"❌ layer3.core failed: {e}")
    import traceback
    traceback.print_exc()
    failed = True

try:
    from src.training.steps.labeling.layer3 import feature_engineering
    print("✅ layer3.feature_engineering imported")
except Exception as e:
    print(f"❌ layer3.feature_engineering failed: {e}")
    import traceback
    traceback.print_exc()
    failed = True

try:
    from src.training.steps.labeling.layer3 import geometry_system
    print("✅ layer3.geometry_system imported")
except Exception as e:
    print(f"❌ layer3.geometry_system failed: {e}")
    import traceback
    traceback.print_exc()
    failed = True

try:
    from src.training.steps.labeling.layer3 import model_training
    print("✅ layer3.model_training imported")
except Exception as e:
    print(f"❌ layer3.model_training failed: {e}")
    import traceback
    traceback.print_exc()
    failed = True

if failed:
    sys.exit(1)
else:
    print("All Layer 3 modules verified.")
