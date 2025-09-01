# Dead Code Detection and Removal

This project includes a comprehensive dead code detection system that automatically runs on GitHub pushes and can be used locally for code cleanup.

## Overview

The dead code detection system uses multiple tools to identify:
- Unused imports
- Unused variables
- Dead functions and classes
- Unreachable code
- Unused type annotations

## Tools Used

### 1. MyPy
- **Purpose**: Static type checking with dead code detection
- **Configuration**: Located in `pyproject.toml` under `[tool.mypy]`
- **Features**:
  - Detects unused imports
  - Identifies unused variables
  - Warns about unreachable code
  - Strict type checking enabled

### 2. Vulture
- **Purpose**: Dead code detection
- **Configuration**: Runs with 80% confidence threshold
- **Features**:
  - Identifies dead functions and classes
  - Detects unused code blocks
  - Excludes test files and generated code

### 3. Ruff
- **Purpose**: Fast Python linter with dead code detection
- **Configuration**: Located in `pyproject.toml` under `[tool.ruff]`
- **Features**:
  - Detects unused imports (F401)
  - Identifies unused variables (F841)
  - Fast execution

### 4. Pyright
- **Purpose**: Microsoft's static type checker
- **Features**:
  - Additional dead code detection
  - Type inference
  - Cross-file analysis

## Usage

### Automatic Detection (GitHub Actions)

The system automatically runs on every push to `main` or `develop` branches. The workflow is defined in `.github/workflows/code_quality.yml`.

### Local Usage

#### Using Makefile (Recommended)

```bash
# Install dependencies
make install

# Run dead code detection (dry run)
make dead-code

# Remove dead code
make dead-code-remove

# Run comprehensive analysis
make dead-code-full

# Run all quality checks
make all

# Quick check (linting + type checking)
make quick
```

#### Using the Dead Code Detector Script

```bash
# Dry run (show what would be removed)
python dead_code_detector.py --dry-run --verbose

# Remove dead code
python dead_code_detector.py --remove --verbose

# Analyze specific directory
python dead_code_detector.py --src-dir src/trading --dry-run

# Save report to file
python dead_code_detector.py --output dead_code_report.md
```

#### Individual Tools

```bash
# MyPy type checking
poetry run mypy src/ --show-error-codes --show-column-numbers

# Vulture dead code detection
poetry run vulture src/ --min-confidence 80 --exclude=test_models,test_results,log

# Ruff unused imports
poetry run ruff check . --select=F401,F841 --output-format=text

# Pyright analysis
poetry run pyright src/
```

## Configuration

### MyPy Configuration

The MyPy configuration in `pyproject.toml` includes:

```toml
[tool.mypy]
python_version = "3.11"
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_return_any = true
warn_unreachable = true
strict_equality = true
warn_unused_configs = true
warn_no_return = true
warn_unused_imports = true
```

### Excluded Directories

The following directories are excluded from dead code analysis:
- `test_models/`
- `test_results/`
- `log/`
- `__pycache__/`
- `.mypy_cache/`
- `.venv/`
- `venv/`

## Understanding Reports

### MyPy Output

MyPy reports include:
- **unused-import**: Imported but never used
- **unused-variable**: Variable assigned but never used
- **unreachable**: Code that can never be executed

### Vulture Output

Vulture reports include:
- Dead functions and classes
- Unused code blocks
- Confidence scores (80%+ threshold)

### Ruff Output

Ruff reports include:
- **F401**: Unused imports
- **F841**: Unused variables

## Best Practices

### Before Removing Code

1. **Review the report**: Always review what will be removed
2. **Check for side effects**: Some imports might have side effects
3. **Verify test coverage**: Ensure tests still pass
4. **Check for dynamic usage**: Some code might be used dynamically

### Safe Removal

```bash
# Always start with dry run
make dead-code

# Review the output
# If satisfied, remove the code
make dead-code-remove

# Run tests to ensure nothing broke
make test
```

### Handling False Positives

Some tools may report false positives. You can:

1. **Add type ignores**: `# type: ignore[unused-import]`
2. **Use `# noqa` comments**: `import unused_module  # noqa: F401`
3. **Exclude specific files**: Add to exclude patterns in configuration

## Integration with CI/CD

The GitHub Actions workflow automatically:
1. Runs all linting tools
2. Performs type checking
3. Detects dead code
4. Reports issues without failing the build

This ensures code quality is maintained without blocking development.

## Troubleshooting

### Common Issues

1. **MyPy cache issues**: Run `make clean` to clear caches
2. **Import errors**: Check if all dependencies are installed
3. **False positives**: Review and add appropriate ignores

### Performance

- Ruff is very fast and runs in seconds
- MyPy can be slower but provides comprehensive analysis
- Vulture is moderate speed with good accuracy

## Contributing

When adding new code:
1. Ensure it's properly typed
2. Remove any unused imports
3. Run `make quick` before committing
4. Address any dead code warnings

## Monitoring

Regular dead code detection helps:
- Reduce codebase size
- Improve maintainability
- Reduce cognitive load
- Speed up development

Run `make dead-code` weekly to keep the codebase clean.