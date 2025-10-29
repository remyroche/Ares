# Poetry Quick Reference for Ares

This document provides quick reference commands for managing the Ares project dependencies with Poetry.

## Quick Start

```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install all dependencies
poetry install

# Activate virtual environment
poetry shell

# Run the application
python ares_launcher.py step01
```

## Essential Commands

### Environment Management

```bash
# Activate virtual environment
poetry shell

# Run command in virtual environment (without activating)
poetry run python script.py

# Show virtual environment path
poetry env info --path
```

### Dependency Management

```bash
# Install all dependencies
poetry install

# Install with development dependencies
poetry install --with=dev

# Install with optional dependencies (additional ML libraries)
poetry install --with=optional

# Install specific group
poetry install --with=dev,optional

# Update all dependencies
poetry update

# Update specific package
poetry update package-name

# Add new dependency
poetry add package-name

# Add development dependency
poetry add --group dev package-name

# Remove dependency
poetry remove package-name
```

### Package Information

```bash
# Show all installed packages
poetry show

# Show dependency tree
poetry show --tree

# Show specific package info
poetry show package-name

# List outdated packages
poetry show --outdated
```

### Project Management

```bash
# Check project configuration
poetry check

# Export requirements.txt
poetry export -f requirements.txt --output requirements.txt --without-hashes

# Build package
poetry build

# Publish package
poetry publish
```

## Dependency Groups

The project uses the following dependency groups:

### Main Dependencies (`main`)
Core required packages for basic functionality:
- numpy, pandas, scipy, scikit-learn
- matplotlib, seaborn, plotly
- tensorflow, torch
- hdbscan, optuna, shap, lime
- vectorbt, ccxt, yfinance

### Development Dependencies (`dev`)
Tools for development and code quality:
- black, isort, flake8, pylint, mypy
- pytest, pytest-asyncio, pytest-cov
- sphinx, rich, ipython, jupyter

### Optional Dependencies (`optional`)
Additional packages for enhanced functionality:
- xgboost, lightgbm, catboost
- statsmodels, arch
- cupy (for GPU acceleration)
- networkx, graphviz

## Platform-Specific Notes

### macOS (M1/M2)
```bash
# The project is optimized for Apple Silicon
poetry install --with=optional

# Verify MPS support
poetry run python -c "import torch; print(torch.backends.mps.is_available())"
```

### Linux (CUDA)
```bash
# For CUDA support
poetry install --with=optional

# Verify CUDA support
poetry run python -c "import torch; print(torch.cuda.is_available())"
```

### Windows
```bash
# Standard installation
poetry install

# For development
poetry install --with=dev
```

## Troubleshooting

### Common Issues

#### Poetry not found
```bash
# Add to PATH (add to ~/.bashrc or ~/.zshrc)
export PATH="$HOME/.local/bin:$PATH"

# Or reinstall Poetry
curl -sSL https://install.python-poetry.org | python3 -
```

#### Dependency conflicts
```bash
# Check for conflicts
poetry show --tree

# Update dependencies
poetry update

# Clear cache and reinstall
poetry cache clear --all pypi
poetry install
```

#### Virtual environment issues
```bash
# Remove existing environment
poetry env remove python

# Recreate environment
poetry install
```

### Performance Optimization

#### Memory management
```bash
# Set environment variables for better memory usage
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_MAX_THREADS=4
```

#### GPU acceleration
```python
# Check available devices
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")
```

## Integration with IDEs

### VS Code
1. Install Python extension
2. Select Poetry interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter"
3. Choose the Poetry virtual environment

### PyCharm
1. Go to Settings → Project → Python Interpreter
2. Add New Interpreter → Poetry Environment
3. Select the project directory

### Jupyter Notebook
```bash
# Install Jupyter in Poetry environment
poetry add --group dev jupyter

# Start Jupyter
poetry run jupyter notebook
```

## Migration from UV/pip

If migrating from UV or pip:

```bash
# Run the migration script
python scripts/migrate_to_poetry.py

# Or manually:
# 1. Backup existing pyproject.toml
# 2. Install Poetry
# 3. Run: poetry install
```

## Best Practices

1. **Always use Poetry commands** instead of pip when possible
2. **Commit poetry.lock** to version control for reproducible builds
3. **Use dependency groups** to separate different types of dependencies
4. **Regularly update** dependencies with `poetry update`
5. **Test after updates** to ensure compatibility

## Advanced Usage

### Custom Scripts
Add custom scripts to `pyproject.toml`:
```toml
[tool.poetry.scripts]
ares = "src.launcher.ares_launcher:main"
test = "pytest"
format = "black src/ && isort src/"
```

Then run with:
```bash
poetry run ares
poetry run test
poetry run format
```

### Environment Variables
Create `.env` file in project root:
```
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
```

### Configuration
Poetry configuration is in `pyproject.toml`:
- Dependencies: `[tool.poetry.dependencies]`
- Dev dependencies: `[tool.poetry.group.dev.dependencies]`
- Optional dependencies: `[tool.poetry.group.optional.dependencies]`
- Scripts: `[tool.poetry.scripts]`
- Build system: `[build-system]`
