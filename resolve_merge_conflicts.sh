#!/bin/bash

# Script to help resolve merge conflicts for the step renaming refactor

echo "Resolving merge conflicts for step renaming refactor..."
echo "============================================"

# List of conflicted files
CONFLICTED_FILES=(
    "src/training/steps/step1_5_data_converter.py"
    "src/training/steps/step1_data_collection.py"
    "src/training/steps/step2_feature_engineering_validator.py"
    "src/training/steps/step3_hmm_regime_discovery_validator.py"
    "src/training/steps/step7_enhanced_matrix_operations_validator.py"
    "src/training/steps/step9_5_multi_timeframe_hmm_ensemble_validator.py"
    "src/utils/hmm_composite_manager.py"
)

# Function to check if file exists with new name
check_renamed_file() {
    local old_file=$1
    local new_file=""
    
    # Determine the new filename
    if [[ $old_file =~ step([1-9])(_|\.py) ]]; then
        new_file=$(echo $old_file | sed -E 's/step([1-9])(_|\.py)/step0\1\2/')
    fi
    
    echo "Checking: $old_file -> $new_file"
    
    if [ -f "$new_file" ]; then
        echo "  ✓ Renamed file exists: $new_file"
        return 0
    else
        echo "  ✗ Renamed file not found"
        return 1
    fi
}

# Resolution strategy
echo -e "\nResolution Strategy:"
echo "===================="
echo "Since we're renaming files from step1-9 to step01-09, the conflicts are likely due to:"
echo "1. The old files being modified in the other branch"
echo "2. Our branch deleting/renaming these files"
echo -e "\nWe need to:"
echo "1. Apply the changes from the other branch to our renamed files"
echo "2. Delete the old filenames"
echo ""

# Check status of each conflicted file
echo "Checking file status..."
echo "======================"
for file in "${CONFLICTED_FILES[@]}"; do
    check_renamed_file "$file"
done

# Manual resolution steps
echo -e "\n\nManual Resolution Steps:"
echo "========================"
echo "1. For each conflicted file, check what changes were made in the other branch:"
echo "   git diff --merge-base HEAD origin/main <filename>"
echo ""
echo "2. Apply those changes to the renamed files:"
echo "   - step1_5_data_converter.py -> step01_5_data_converter.py"
echo "   - step1_data_collection.py -> step01_data_collection.py"
echo "   - step2_feature_engineering_validator.py -> step02_feature_engineering_validator.py"
echo "   - step3_hmm_regime_discovery_validator.py -> step03_hmm_regime_discovery_validator.py"
echo "   - step7_enhanced_matrix_operations_validator.py -> step07_enhanced_matrix_operations_validator.py"
echo "   - step9_5_multi_timeframe_hmm_ensemble_validator.py -> step09_5_multi_timeframe_hmm_ensemble_validator.py"
echo ""
echo "3. For src/utils/hmm_composite_manager.py:"
echo "   - This file likely has import statements that need updating"
echo "   - Change imports from step1-9 to step01-09"
echo ""
echo "4. After applying changes, remove the old files and add the new ones:"
echo "   git rm src/training/steps/step[1-9]*.py"
echo "   git add src/training/steps/step0[1-9]*.py"
echo "   git add src/utils/hmm_composite_manager.py"
echo ""
echo "5. Complete the merge:"
echo "   git commit -m 'resolve: merge conflicts after step renumbering'"
echo ""

# Automated resolution attempt (optional)
echo -e "\nAutomated Resolution Commands:"
echo "=============================="
echo "# First, let's see the actual conflicts"
echo "git status --porcelain | grep '^UU'"
echo ""
echo "# For each step file conflict, the resolution is to use our renamed version"
echo "# and delete the old filename:"
echo ""
for file in "${CONFLICTED_FILES[@]}"; do
    if [[ $file =~ src/training/steps/step[1-9] ]]; then
        new_file=$(echo $file | sed -E 's/step([1-9])(_|\.py)/step0\1\2/')
        echo "# For $file:"
        echo "git rm $file"
        echo "git add $new_file"
        echo ""
    fi
done

echo "# For hmm_composite_manager.py, we need to manually merge and update imports"
echo "# Open the file and resolve conflicts, ensuring all imports use step01-09 format"
echo ""

echo -e "\nTo create a new PR after resolving conflicts:"
echo "============================================="
echo "1. git add -A"
echo "2. git commit -m 'resolve: merge conflicts for step renumbering refactor'"
echo "3. git push origin <your-branch-name>"
echo "4. Create a new PR on GitHub"