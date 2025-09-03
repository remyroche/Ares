# Merge Conflict Resolution Guide

## Understanding the Conflicts

The conflicts are occurring because:
1. **Our branch**: Renamed files from `step1-9` to `step01-09` format
2. **Main branch**: Made changes to the original `step1-9` files
3. Git sees this as: "You deleted files that were modified in the other branch"

## Conflicted Files

### Step Files (renamed in our branch)
- `step1_5_data_converter.py` → `step01_5_data_converter.py`
- `step1_data_collection.py` → `step01_data_collection.py`
- `step2_feature_engineering_validator.py` → `step02_feature_engineering_validator.py`
- `step3_hmm_regime_discovery_validator.py` → `step03_hmm_regime_discovery_validator.py`
- `step7_enhanced_matrix_operations_validator.py` → `step07_enhanced_matrix_operations_validator.py`
- `step9_5_multi_timeframe_hmm_ensemble_validator.py` → `step09_5_multi_timeframe_hmm_ensemble_validator.py`

### Utility File (needs import updates)
- `src/utils/hmm_composite_manager.py`

## Resolution Steps

### Step 1: Check what changed in main branch
```bash
# See what changes were made to each file in main
git diff origin/main...HEAD -- src/training/steps/step1_5_data_converter.py
git diff origin/main...HEAD -- src/training/steps/step1_data_collection.py
# ... repeat for each conflicted file
```

### Step 2: Apply main branch changes to renamed files

For each conflicted step file:

1. **View the changes from main branch**:
   ```bash
   git show origin/main:src/training/steps/step1_5_data_converter.py > temp_main_version.py
   ```

2. **Compare with your renamed file**:
   ```bash
   diff temp_main_version.py src/training/steps/step01_5_data_converter.py
   ```

3. **Apply any new changes from main to the renamed file**:
   - Open `src/training/steps/step01_5_data_converter.py`
   - Manually apply any changes that were made in main branch
   - Ensure all imports still use the new `step01-09` format

### Step 3: Resolve hmm_composite_manager.py

This file likely has import conflicts. Open it and:
1. Look for conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`)
2. Update all imports to use `step01-09` format
3. Keep any other changes from main branch

Example resolution:
```python
# Before (main branch)
from src.training.steps.step3_hmm_regime_discovery import ...

# After (your resolution)
from src.training.steps.step03_hmm_regime_discovery import ...
```

### Step 4: Complete the resolution

```bash
# Remove the old-named files (they're now renamed)
git rm src/training/steps/step1_5_data_converter.py
git rm src/training/steps/step1_data_collection.py
git rm src/training/steps/step2_feature_engineering_validator.py
git rm src/training/steps/step3_hmm_regime_discovery_validator.py
git rm src/training/steps/step7_enhanced_matrix_operations_validator.py
git rm src/training/steps/step9_5_multi_timeframe_hmm_ensemble_validator.py

# Add the renamed versions
git add src/training/steps/step01_5_data_converter.py
git add src/training/steps/step01_data_collection.py
git add src/training/steps/step02_feature_engineering_validator.py
git add src/training/steps/step03_hmm_regime_discovery_validator.py
git add src/training/steps/step07_enhanced_matrix_operations_validator.py
git add src/training/steps/step09_5_multi_timeframe_hmm_ensemble_validator.py

# Add the resolved utility file
git add src/utils/hmm_composite_manager.py

# Commit the resolution
git commit -m "resolve: merge conflicts after step renumbering

- Applied changes from main to renamed step files
- Updated imports in hmm_composite_manager.py
- Maintained zero-padded naming convention (step01-09)"
```

### Step 5: Push and create new PR

```bash
# Push to your branch
git push origin your-branch-name

# Go to GitHub and create a new PR
```

## Important Notes

1. **Don't lose main branch changes**: Make sure any functionality changes from main are preserved in the renamed files
2. **Maintain naming consistency**: All imports should use `step01-09` format
3. **Test after resolution**: Run any tests to ensure nothing broke during conflict resolution