import os
from pathlib import Path

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

def fix_file(file_path):
    print(f"Processing {file_path}...")
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    skip_until_next_try = False
    in_corrupted_block = False
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Detect the corrupted block start
        if "standardized_output = self._create_standardized_output(" in line:
            # We found the start. We should take the next few lines until the closing parenthesis
            new_lines.append(line)
            i += 1
            while i < len(lines) and ")" not in lines[i]:
                new_lines.append(lines[i])
                i += 1
            if i < len(lines):
                new_lines.append(lines[i]) # The line with ")"
                i += 1
            
            # Now, skip any immediately following lines that look like "else (features_df..."
            while i < len(lines) and ("else (features_df" in lines[i] or "full_labels," in lines[i] or "final_preds.values" in lines[i]):
                print(f"  Skipping corrupted line: {lines[i].strip()}")
                i += 1
            continue
            
        # Detect duplicated AFML Audit comment
        if "# AFML Audit: Update metrics using full OOF set" in line:
            new_lines.append(line)
            i += 1
            while i < len(lines) and "# AFML Audit: Update metrics using full OOF set" in lines[i]:
                print(f"  Skipping duplicated comment: {lines[i].strip()}")
                i += 1
            continue

        new_lines.append(line)
        i += 1
        
    with open(file_path, 'w') as f:
        f.writelines(new_lines)
    print(f"✅ Fixed {file_path}")

for f_rel in specialist_files:
    f_path = Path(f_rel)
    if f_path.exists():
        fix_file(f_path)
    else:
        print(f"❌ File not found: {f_rel}")
