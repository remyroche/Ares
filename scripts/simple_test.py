#!/usr/bin/env python3
"""
Simple test of MDA/SHAP import.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.training.steps.labeling.mda_shap_feature_selection import MDA_SHAP_FeatureSelector
    print("✅ Import successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()







