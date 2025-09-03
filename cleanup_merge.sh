#!/bin/bash
cd /workspace

# Remove merge-related files
rm -f .git/MERGE_HEAD
rm -f .git/MERGE_MSG
rm -f .git/MERGE_MODE
rm -f .git/AUTO_MERGE
rm -f .git/.MERGE_MSG.swp

echo "Merge files cleaned up"

# Add all changes
git add -A

# Create commit
git commit -m "Fix syntax errors in multiple Python files

- Fixed decorator syntax issues (missing parentheses)
- Corrected import statement placement
- Fixed indentation errors
- Resolved unterminated string literals
- Fixed unmatched parentheses in multiple files

Files fixed:
- model_trainer.py
- step_orchestrator.py
- integration_guide.py
- binance.py
- model_manager.py
- Multiple validation step files
- config.py
- paper_trader.py"

echo "Commit created"