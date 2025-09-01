# Makefile for Code Quality and Dead Code Detection

.PHONY: help install test lint typecheck dead-code clean

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install dependencies"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting tools"
	@echo "  typecheck    - Run MyPy type checking"
	@echo "  dead-code    - Detect dead code (dry run)"
	@echo "  dead-code-remove - Remove dead code"
	@echo "  clean        - Clean up cache files"
	@echo "  all          - Run all quality checks"

# Install dependencies
install:
	python3 -m venv venv
	venv/bin/pip install mypy vulture ruff pyright

# Run tests
test:
	venv/bin/python -m pytest

# Run linting tools
lint:
	venv/bin/ruff check .
	venv/bin/ruff format . --check

# Run type checking
typecheck:
	venv/bin/mypy src/ --show-error-codes --show-column-numbers

# Detect dead code (dry run)
dead-code:
	@echo "Running dead code detection (dry run)..."
	venv/bin/python dead_code_detector.py --dry-run --verbose

# Remove dead code
dead-code-remove:
	@echo "Running dead code detection and removal..."
	venv/bin/python dead_code_detector.py --remove --verbose

# Run vulture specifically
vulture:
	@echo "Running Vulture dead code detection..."
	venv/bin/vulture src/ --min-confidence 80 --exclude=test_models,test_results,log

# Run comprehensive dead code analysis
dead-code-full:
	@echo "=== Comprehensive Dead Code Analysis ==="
	@echo "1. Running MyPy..."
	venv/bin/mypy src/ --show-error-codes --show-column-numbers || true
	@echo ""
	@echo "2. Running Vulture..."
	venv/bin/vulture src/ --min-confidence 80 --exclude=test_models,test_results,log || true
	@echo ""
	@echo "3. Running Ruff for unused imports..."
	venv/bin/ruff check . --select=F401,F841 --output-format=text || true
	@echo ""
	@echo "4. Running custom dead code detector..."
	venv/bin/python dead_code_detector.py --dry-run

# Clean up cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

# Run all quality checks
all: lint typecheck dead-code-full

# Quick check (just linting and type checking)
quick: lint typecheck