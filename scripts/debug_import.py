
import sys
from pathlib import Path

# Ensure project root is in python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

print("🔍 Attempting to import TrainSpecialistsWithGMMStep...")
try:
    from src.training.steps.labeling.train_specialists_with_gmm_step import TrainSpecialistsWithGMMStep
    print("✅ Successfully imported TrainSpecialistsWithGMMStep")
except Exception as e:
    print(f"❌ Failed to import TrainSpecialistsWithGMMStep: {e}")
    import traceback
    traceback.print_exc()
