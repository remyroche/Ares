#!/bin/bash
# Commands to create a PR for the step renaming changes

# 1. Check current status
git status

# 2. Add all renamed files and changes
git add -A

# 3. Commit with a descriptive message
git commit -m "refactor: standardize step numbering with zero-padding

- Renamed all single-digit steps (step1-9) to zero-padded format (step01-09)
- Updated all imports across 43+ Python files
- Updated references in YAML/JSON configuration files
- Consolidated duplicate functionality:
  - Removed step8_regime_data_splitting (kept step4)
  - Consolidated feature engineering into vectorized version
  - Made enhanced HMM training the main implementation
- Cleaned up refactored file versions
- Improved overall pipeline organization and consistency"

# 4. Push to a new branch
git checkout -b refactor/standardize-step-numbering
git push origin refactor/standardize-step-numbering

# 5. Create PR via GitHub CLI (if installed)
# gh pr create --title "Refactor: Standardize training step numbering" \
#   --body "## Summary
# 
# This PR standardizes the training pipeline step numbering by adding zero-padding to single-digit steps and cleaning up duplicate/refactored files.
# 
# ## Changes Made
# 
# ### 1. Step Numbering Standardization
# - Renamed all single-digit steps (step1-9) to zero-padded format (step01-09)
# - Updated all imports and references throughout the codebase
# - Affected 33 step files and 43+ Python files with imports
# 
# ### 2. File Consolidation
# - Removed duplicate regime_data_splitting implementations (kept step04)
# - Consolidated feature engineering into vectorized version (step06)
# - Made enhanced HMM training the main implementation (step09)
# - Cleaned up refactored file versions
# 
# ### 3. Pipeline Organization
# - Renamed step4_triple_barrier_method to step04_5 for logical flow
# - Fixed __init__.py references to match actual file names
# - Updated configuration files with new step names
# 
# ## Impact
# - No functional changes, only file organization
# - All imports and references have been updated
# - Pipeline flow is now more logical and consistent"