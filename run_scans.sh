#!/bin/bash

# run_scans.sh
# This script runs static analysis tools to check for code quality and security vulnerabilities.
# Ensure you have the necessary tools installed:
# pip install ruff bandit mypy

echo "--- Running Ruff (Linter and Formatter) ---"

# To automatically fix issues, you could use:
ruff format .
ruff check . --fix

# Check for linting errors and formatting issues without fixing them
#ruff check .

echo ""
echo "--- Running MyPy (Static Type Checker) ---"
# Check for type consistency throughout the project.
# --ignore-missing-imports is useful to prevent errors from libraries that don't have type stubs.
# mypy . --ignore-missing-imports

# For even stricter checking in the future, you could consider:
mypy . --strict

echo ""
echo "Scan complete."
