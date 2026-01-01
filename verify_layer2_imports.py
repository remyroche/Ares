import sys
import os

# Add src to path
sys.path.append(os.getcwd())

print("Testing imports...")
try:
    from src.training.steps.labeling.label_based_layer_2 import LabelBasedLayer2
    print("✅ LabelBasedLayer2 imported successfully")
except Exception as e:
    print(f"❌ Failed to import LabelBasedLayer2: {e}")
    import traceback
    traceback.print_exc()

try:
    import catboost
    print(f"✅ CatBoost available: {catboost.__version__}")
except ImportError:
    print("⚠️ CatBoost NOT installed")

try:
    from src.training.steps.labeling.focal_loss_utils import get_focal_loss_lgbm, get_focal_loss_xgb, RobustFocalLoss
    print("✅ Focal Loss Utils imported successfully")
    # Instantation check
    rf = RobustFocalLoss(gamma_pos=1.0, gamma_neg=2.0)
    print("✅ RobustFocalLoss instantiated successfully")
except Exception as e:
    print(f"❌ Failed to import/init Focal Loss Utils: {e}")

print("Verification complete.")
