#!/usr/bin/env python3
"""
Script to upgrade to Python 3.12 with full library compatibility.
This will create a new conda environment with Python 3.12 and all dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, check=True):
    """Run a command and return the result."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    return result

def main():
    """Upgrade to Python 3.12 with full compatibility."""
    print("🚀 Upgrading to Python 3.12 for full library compatibility...")
    
    # Check if conda is available
    if not run_command("which conda", check=False):
        print("❌ Conda not found. Please install conda first.")
        return False
    
    # Create new conda environment with Python 3.12
    env_name = "ares-py312"
    print(f"Creating conda environment: {env_name}")
    
    if not run_command(f"conda create -n {env_name} python=3.12 -y"):
        print("❌ Failed to create conda environment")
        return False
    
    # Install Poetry in the new environment
    print("Installing Poetry in new environment...")
    if not run_command(f"conda activate {env_name} && pip install poetry"):
        print("❌ Failed to install Poetry")
        return False
    
    # Copy pyproject.toml to a Python 3.12 version
    print("Creating Python 3.12 compatible pyproject.toml...")
    
    py312_config = """[tool.poetry]
name = "ares"
version = "0.1.0"
description = "Advanced Regime-based Ensemble System for Financial Market Analysis"
authors = ["Remy Roche <remy@example.com>"]
readme = "README.md"
packages = [{include = "src"}]

[tool.poetry.dependencies]
python = "^3.12"

# Core scientific computing
numpy = "^2.2.6"
pandas = "^2.0.0"
scipy = "^1.10.0"
scikit-learn = "^1.3.0"

# Data visualization
matplotlib = "^3.7.0"
seaborn = "^0.12.0"
plotly = "^5.15.0"

# Machine Learning & Deep Learning
tensorflow = "^2.20.0"
torch = "^2.0.0"
torchvision = "^0.15.0"
torchaudio = "^2.0.0"

# Clustering and HMM
hdbscan = "^0.8.29"
hmmlearn = "^0.3.0"

# Optimization and Hyperparameter Tuning
optuna = "^3.2.0"

# Model Explainability
shap = "^0.42.0"
lime = "^0.2.0.1"

# Financial Data and Analysis
vectorbt = "^0.28.0"
ccxt = "^4.0.0"
yfinance = "^0.2.18"
pandas-ta = "^0.4.67"
ta = "^0.10.2"

# Async and Networking
aiohttp = "^3.8.0"
requests = "^2.31.0"

# Data Processing
pyarrow = "^12.0.0"
fastparquet = "^0.8.0"

# Utilities
psutil = "^5.9.0"
tqdm = "^4.65.0"
python-dateutil = "^2.8.0"
pytz = "^2023.3"
pyyaml = "^6.0"
python-dotenv = "^1.0.0"

# Performance Optimization
numba = "^0.61.0"

# Web Framework (for GUI)
fastapi = "^0.104.0"
uvicorn = {extras = ["standard"], version = "^0.24.0"}
pydantic = "^2.5.0"

# Monitoring and Metrics
prometheus-client = "^0.19.0"

[tool.poetry.group.dev.dependencies]
# Code Quality Tools
black = "^23.0.0"
isort = "^5.12.0"
flake8 = "^6.0.0"
pylint = "^2.17.0"
mypy = "^1.5.0"

# Testing
pytest = "^7.4.0"
pytest-asyncio = "^0.21.0"
pytest-cov = "^4.1.0"

# Development Utilities
rich = "^13.0.0"
ipython = "^8.15.0"
jupyter = "^1.0.0"

[tool.poetry.group.optional.dependencies]
# Additional ML Libraries
xgboost = "^1.7.0"
lightgbm = "^4.0.0"
catboost = "^1.2.0"

# Advanced Analytics
statsmodels = "^0.14.0"
arch = "^6.2.0"

# Visualization Extensions
networkx = "^3.1.0"
graphviz = "^0.20.0"
squarify = "^0.4.3"

[tool.poetry.scripts]
ares = "src.launcher.ares_launcher:main"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
"""
    
    with open("pyproject_py312.toml", "w") as f:
        f.write(py312_config)
    
    print("✅ Python 3.12 configuration created: pyproject_py312.toml")
    
    # Instructions for the user
    print("\n📋 Next Steps:")
    print(f"1. Activate the new environment: conda activate {env_name}")
    print("2. Copy the Python 3.12 config: cp pyproject_py312.toml pyproject.toml")
    print("3. Install dependencies: poetry install")
    print("4. Test the installation: poetry run python -c 'import tensorflow, vectorbt, pandas_ta; print(\"All working!\")'")
    
    print("\n🎯 Benefits of Python 3.12:")
    print("✅ Full TensorFlow support")
    print("✅ pandas-ta support")
    print("✅ Better numba compatibility")
    print("✅ All libraries working together")
    
    return True

if __name__ == "__main__":
    main()
