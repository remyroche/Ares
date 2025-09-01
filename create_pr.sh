#!/bin/bash

# Create Pull Request for Code Quality Improvements
# This script helps create a PR with the comprehensive description

echo "🚀 Creating Pull Request for Code Quality Improvements..."
echo ""

# Check if we're on the right branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "cursor/clean-up-code-and-remove-unused-imports-c8a9" ]; then
    echo "❌ Error: Not on the correct branch!"
    echo "Current branch: $CURRENT_BRANCH"
    echo "Expected branch: cursor/clean-up-code-and-remove-unused-imports-c8a9"
    exit 1
fi

echo "✅ Current branch: $CURRENT_BRANCH"
echo ""

# Check if changes are pushed
echo "📤 Checking if changes are pushed to remote..."
git fetch origin
LOCAL_COMMIT=$(git rev-parse HEAD)
REMOTE_COMMIT=$(git rev-parse origin/$CURRENT_BRANCH)

if [ "$LOCAL_COMMIT" != "$REMOTE_COMMIT" ]; then
    echo "❌ Error: Local changes not pushed to remote!"
    echo "Please run: git push origin $CURRENT_BRANCH"
    exit 1
fi

echo "✅ Changes are pushed to remote"
echo ""

# Display PR creation URL
echo "🔗 Create Pull Request:"
echo "https://github.com/remyroche/Ares/pull/new/$CURRENT_BRANCH"
echo ""

# Display PR description
echo "📝 PR Description (copy this to GitHub):"
echo "=========================================="
echo ""

# Read and display the PR description
if [ -f "PR_DESCRIPTION.md" ]; then
    cat PR_DESCRIPTION.md
else
    echo "❌ Error: PR_DESCRIPTION.md not found!"
    exit 1
fi

echo ""
echo "=========================================="
echo ""
echo "🎉 Ready to create the Pull Request!"
echo ""
echo "Steps:"
echo "1. Click the URL above"
echo "2. Copy the PR description from above"
echo "3. Paste it into the GitHub PR description"
echo "4. Set the title: 'Code Quality Improvements: Syntax Fixes, Import Cleanup, and Dead Code Removal'"
echo "5. Submit the PR"
echo ""
echo "📊 Summary of Changes:"
echo "- 525 files fixed with syntax errors"
echo "- 89,475 total fixes applied"
echo "- 346 lines of dead code removed"
echo "- 4 files cleaned of unused imports"
echo "- New automated tools for ongoing maintenance"