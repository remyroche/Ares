# Ares Installation Guide

This guide covers installing the Ares project dependencies using Poetry for optimal dependency management and version compatibility.

## Prerequisites

- Python 3.11 or higher
- pip (Python package installer)
- Git (for cloning the repository)

## Installation Methods

### Method 1: Poetry (Recommended)

Poetry provides the best dependency management with automatic conflict resolution and virtual environment handling.

#### 1. Install Poetry

**macOS/Linux:**
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

**Windows (PowerShell):**
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

**Alternative (using pip):**
```bash
pip install poetry
```

#### 2. Install Dependencies

```bash
# Install all dependencies
poetry install

# Install with development dependencies
poetry install --with=dev

# Install specific groups
poetry install --with=optional  # For additional ML libraries
```

#### 3. Activate Virtual Environment

```bash
poetry shell
```

#### 4. Run the Application

```bash
# Using Poetry
poetry run python ares_launcher.py step01

# Or activate shell first
poetry shell
python ares_launcher.py step01
```

### Method 2: pip with requirements.txt

If you prefer using pip directly:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Platform-Specific Optimizations

### macOS (M1/M2 Macs)

The project is optimized for Apple Silicon Macs with the following features:
- Metal Performance Shaders (MPS) support for PyTorch
- Optimized NumPy and SciPy builds
- Memory-efficient data processing

```bash
# Install with M1 optimizations
poetry install --with=optional
```

### Linux (CUDA Support)

For CUDA-enabled systems:

```bash
# Install CUDA-optimized PyTorch
poetry install --with=optional
```

### Windows

Standard installation works on Windows. For GPU acceleration, ensure you have:
- CUDA toolkit installed
- Compatible NVIDIA drivers

## Dependency Management

### Using the Dependency Manager Script

The project includes a comprehensive dependency management script:

```bash
# Check compatibility
python scripts/manage_dependencies.py check

# Install dependencies
python scripts/manage_dependencies.py install

# Install with development tools
python scripts/manage_dependencies.py install --dev

# Update dependencies
python scripts/manage_dependencies.py update

# Check for conflicts
python scripts/manage_dependencies.py conflicts

# Export requirements
python scripts/manage_dependencies.py export
```

### Common Commands

```bash
# Show installed packages
poetry show

# Show dependency tree
poetry show --tree

# Add new dependency
poetry add package-name

# Add development dependency
poetry add --group dev package-name

# Update specific package
poetry update package-name

# Remove package
poetry remove package-name
```

## Troubleshooting

### Common Issues

#### 1. Python Version Compatibility

Ensure you're using Python 3.11 or higher:
```bash
python --version
```

#### 2. Poetry Not Found

If Poetry is not recognized, add it to your PATH:
```bash
# Add to ~/.bashrc or ~/.zshrc
export PATH="$HOME/.local/bin:$PATH"
```

#### 3. Dependency Conflicts

Check for conflicts:
```bash
python scripts/manage_dependencies.py conflicts
```

#### 4. M1 Mac Issues

For M1 Macs, ensure you're using the correct architecture:
```bash
# Check architecture
uname -m

# Should show arm64 for M1/M2
```

#### 5. CUDA Issues

For CUDA-related problems:
```bash
# Check CUDA installation
nvidia-smi

# Verify PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"
```

### Performance Optimization

#### Memory Management

For large datasets, consider:
```bash
# Set memory limits
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

#### GPU Acceleration

Enable GPU acceleration where available:
```python
# In your code
import torch
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
```

## Development Setup

### Pre-commit Hooks

Install pre-commit hooks for code quality:
```bash
poetry run pre-commit install
```

### Code Formatting

Format code using Black and isort:
```bash
poetry run black src/
poetry run isort src/
```

### Type Checking

Run type checking with mypy:
```bash
poetry run mypy src/
```

### Testing

Run tests:
```bash
poetry run pytest
```

## Verification

After installation, verify everything works:

```bash
# Run compatibility check
python scripts/manage_dependencies.py check

# Test basic functionality
python -c "import numpy, pandas, sklearn, torch; print('All core dependencies working!')"

# Test GPU support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Run the compatibility check: `python scripts/manage_dependencies.py check`
3. Check the logs for specific error messages
4. Ensure all prerequisites are met

## Dependencies Overview

### Core Dependencies
- **numpy**: Numerical computing
- **pandas**: Data manipulation
- **scipy**: Scientific computing
- **scikit-learn**: Machine learning

### Machine Learning
- **tensorflow**: Deep learning framework
- **torch**: PyTorch deep learning
- **hdbscan**: Clustering algorithms
- **optuna**: Hyperparameter optimization

### Financial Analysis
- **vectorbt**: Vectorized backtesting
- **ccxt**: Cryptocurrency exchange APIs
- **yfinance**: Financial data
- **pandas-ta**: Technical analysis

### Visualization
- **matplotlib**: Plotting
- **seaborn**: Statistical visualization
- **plotly**: Interactive plots

### Utilities
- **psutil**: System monitoring
- **tqdm**: Progress bars
- **pyyaml**: Configuration files
- **aiohttp**: Async HTTP client
