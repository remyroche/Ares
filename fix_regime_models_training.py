#!/usr/bin/env python3
"""
Script to clean up duplicate/unreachable code in regime_models_training.py
"""

def fix_regime_models_training():
    """Remove unreachable code from regime_models_training.py"""
    
    file_path = "/Users/remyroche/Documents/Ares/src/training/steps/market_analysis/components/regime_models_training.py"
    
    # Read the file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find the line where the execute method ends (around line 1091)
    # And find where the second _generate_regime_probability_report starts (around line 1525)
    
    # Strategy: Keep everything up to and including line 1091 (end of execute method)
    # Then skip to line 1525 (start of correct _generate_regime_probability_report)
    
    # Look for the pattern to identify the end of the execute method
    execute_end_idx = None
    for i, line in enumerate(lines):
        if i > 1085 and i < 1095:
            if "metadata={'component_type': 'regime_models_training'}" in line and lines[i+1].strip() == ")":
                execute_end_idx = i + 2  # Include the closing paren and blank line
                break
    
    # Look for the second occurrence of "_generate_regime_probability_report" method
    second_method_idx = None
    method_count = 0
    for i, line in enumerate(lines):
        if "async def _generate_regime_probability_report(" in line:
            method_count += 1
            if method_count == 2:
                second_method_idx = i
                break
    
    if execute_end_idx and second_method_idx:
        print(f"Found execute method end at line {execute_end_idx + 1}")
        print(f"Found second _generate_regime_probability_report at line {second_method_idx + 1}")
        print(f"Removing {second_method_idx - execute_end_idx} lines of unreachable code")
        
        # Create new content: before + after (skipping the unreachable middle section)
        new_lines = lines[:execute_end_idx] + lines[second_method_idx:]
        
        # Write back
        with open(file_path, 'w') as f:
            f.writelines(new_lines)
        
        print(f"✅ Successfully cleaned up the file")
        print(f"   Original lines: {len(lines)}")
        print(f"   New lines: {len(new_lines)}")
        print(f"   Removed: {len(lines) - len(new_lines)} lines")
    else:
        print(f"❌ Could not find the markers:")
        print(f"   execute_end_idx: {execute_end_idx}")
        print(f"   second_method_idx: {second_method_idx}")

if __name__ == '__main__':
    fix_regime_models_training()

