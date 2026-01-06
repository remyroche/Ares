
import os
import re
from pathlib import Path

def clean_file(file_path):
    print(f"Cleaning {file_path}...")
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Pattern to find the corrupted block in execute()
    # It usually looks like:
    # standardized_output = self._create_standardized_output(...)
    # )
    # else (features_df ...),
    # ...
    # )
    
    # Fix the duplicated standardized_output blocks
    pattern = r'(standardized_output = self\._create_standardized_output\(.*?\n\s+\))(\s+else \(features_df if \'features_df\' in locals\(\) else X\),.*?\n\s+\))+'
    
    new_content = re.sub(pattern, r'\1', content, flags=re.DOTALL)
    
    # Also fix any remaining "else (features_df..." lines that might be floating
    new_content = re.sub(r'^\s+else \(features_df if \'features_df\' in locals\(\) else X\),.*?\n', '', new_content, flags=re.MULTILINE)
    
    # Fix duplicated "AFML Audit: Update metrics using full OOF set"
    new_content = re.sub(r'(# AFML Audit: Update metrics using full OOF set\s+)+', r'\1', new_content)
    
    if new_content != content:
        with open(file_path, 'w') as f:
            f.write(new_content)
        print(f"✅ Fixed {file_path}")
    else:
        print(f"ℹ️ No corruption found in {file_path}")

specialist_files = [
    "src/training/steps/market_analysis/ml_liquidity_regime_step_enhanced.py",
    "src/training/steps/market_analysis/ml_path_regime_step_enhanced.py",
    "src/training/steps/market_analysis/ml_reversion_regime_step_enhanced.py",
    "src/training/steps/market_analysis/ml_risk_regime_step_enhanced.py",
    "src/training/steps/market_analysis/ml_smc_regime_step_enhanced.py",
    "src/training/steps/market_analysis/ml_volume_force_step_enhanced.py",
    "src/training/steps/market_analysis/xgb_macro_regime_step_enhanced.py",
    "src/training/steps/market_analysis/xgb_meso_regime_step_enhanced.py"
]

for f_rel in specialist_files:
    f_path = Path(f_rel)
    if f_path.exists():
        clean_file(f_path)
    else:
        print(f"❌ File not found: {f_rel}")
