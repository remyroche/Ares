import os
import re
from pathlib import Path

specialist_files = [
    "src/training/steps/market_analysis/ml_liquidity_regime_step_enhanced.py",
    "src/training/steps/market_analysis/ml_path_regime_step_enhanced.py",
    "src/training/steps/market_analysis/ml_reversion_regime_step_enhanced.py",
    "src/training/steps/market_analysis/ml_risk_regime_step_enhanced.py",
    "src/training/steps/market_analysis/ml_smc_regime_step_enhanced.py",
    "src/training/steps/market_analysis/ml_volume_force_step_enhanced.py",
    "src/training/steps/market_analysis/xgb_macro_regime_step_enhanced.py",
    "src/training/steps/market_analysis/xgb_meso_regime_step_enhanced.py",
    "src/training/steps/market_analysis/ml_volatility_burst_step_enhanced.py"
]

def fix_specialist(file_path):
    print(f"Repairing {file_path}...")
    if not os.path.exists(file_path):
        print(f"  File not found: {file_path}")
        return

    with open(file_path, 'r') as f:
        content = f.read()

    # 1. Fix tprint import
    if "from src.utils.tprint import (" in content:
        if "    tprint," not in content:
            content = content.replace("from src.utils.tprint import (", "from src.utils.tprint import (\n    tprint,")
    elif "from src.utils.tprint import tprint_info" in content:
        content = content.replace("from src.utils.tprint import tprint_info", "from src.utils.tprint import tprint, tprint_info")

    # 2. Fix MIOptimizedFeaturePipeline import if missing
    if "MIOptimizedFeaturePipeline()" in content and "MIOptimizedFeaturePipeline" not in content.split("class")[0]:
        content = "from src.training.steps.market_analysis.enhanced_feature_generators import MIOptimizedFeaturePipeline\n" + content

    # 3. Fix _create_standardized_output call site
    # Replace the broken call with a full one
    broken_call_pattern = r"standardized_output = self\._create_standardized_output\(\s+(feature_df if 'feature_df' in locals\(\) else \(features_df if 'features_df' in locals\(\) else X\)),?\s+\)"
    replacement_call = """standardized_output = self._create_standardized_output(
                feature_df if 'feature_df' in locals() else (features_df if 'features_df' in locals() else X), 
                full_labels, final_preds.values, final_probs.values, symbol, exchange, timeframe, direction
            )"""
    
    content = re.sub(broken_call_pattern, replacement_call, content)

    # 4. Final safety check on standardized_output call (sometimes it's multi-line)
    # If it still has only 1 arg inside self._create_standardized_output(...), fix it.
    
    # Write back
    with open(file_path, 'w') as f:
        f.write(content)
    print(f"  Successfully repaired {file_path}")

for f in specialist_files:
    fix_specialist(f)

print("Done.")
