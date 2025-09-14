#!/usr/bin/env python3
"""
Debug script to extract detailed validation information from the current session.
"""

import sys
import os
sys.path.append('/Users/remyroche/Documents/Ares/src')

from src.utils.tprint import tprint
from src.training.utils.feature_selection.main_framework import FeatureSelectionFramework
import numpy as np
import pandas as pd

def main():
    """Extract and display detailed validation information."""
    tprint("🔍 Extracting detailed validation information...")

    try:
        # Initialize the framework
        framework = FeatureSelectionFramework()

        # Check if there's cached validation data
        cache = framework.data_validator
        tprint("✅ DataValidator initialized")

        # Try to get the most recent validation results from cache
        # Since we can't access the exact data that was validated, let's show what we know

        tprint("\n📊 KNOWN VALIDATION ISSUES FROM LOGS:")
        tprint("==========================================")
        tprint("• Constant features: 1 feature (index 22)")
        tprint("• Zero variance features: 1 feature (index 22)")
        tprint("• Highly correlated feature pairs: 40 pairs")
        tprint("• Perfectly correlated feature pairs: 15 pairs")
        tprint("• Data distribution issues: 111 issues")

        tprint("\n⚠️  NOTE: Specific feature names require access to the original DataFrame")
        tprint("   that was validated. The validation methods return indices, not names.")

        # Show what information we can provide
        tprint("\n🔧 VALIDATION METHODS AVAILABLE:")
        tprint("=================================")
        methods = [
            'detect_constant_features',
            'detect_zero_variance_features',
            'detect_high_correlation_features',
            'detect_perfect_correlations',
            'detect_nan_inf_features'
        ]

        for method in methods:
            if hasattr(cache, method):
                tprint(f"✅ {method}")

        tprint("\n💡 To get specific feature names, you would need:")
        tprint("   1. The original DataFrame with feature names")
        tprint("   2. Run validation on that DataFrame")
        tprint("   3. Map the returned indices back to feature names")

    except Exception as e:
        tprint(f"❌ Error extracting validation details: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
