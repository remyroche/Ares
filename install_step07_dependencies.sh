#!/bin/bash
# Step07 Dependency Installation Script

echo "🚀 Step07 Dependency Installation"
echo "=================================="

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
    PIP_CMD="pip"
elif command -v conda &> /dev/null; then
    echo "✅ Conda detected"
    PIP_CMD="conda install -c conda-forge"
else
    echo "⚠️ No virtual environment detected"
    echo "Creating virtual environment..."
    python3 -m venv step07_env
    source step07_env/bin/activate
    PIP_CMD="pip"
fi

echo "📦 Installing dependencies..."

# Install core dependencies
$PIP_CMD install numpy>=1.21.0
$PIP_CMD install pandas>=1.3.0
$PIP_CMD install scikit-learn>=1.0.0
$PIP_CMD install scipy>=1.7.0
$PIP_CMD install psutil>=5.8.0

# Install optional dependencies
echo "📦 Installing optional dependencies..."
$PIP_CMD install torch>=1.12.0 || echo "⚠️ PyTorch installation failed, continuing..."
$PIP_CMD install numba>=0.56.0 || echo "⚠️ Numba installation failed, continuing..."
$PIP_CMD install lightgbm>=3.3.0 || echo "⚠️ LightGBM installation failed, continuing..."

echo "🧪 Testing installation..."
python3 -c "
import numpy as np
import pandas as pd
import sklearn
import scipy
import psutil
print('✅ Core dependencies working')

try:
    import torch
    print('✅ PyTorch working')
except ImportError:
    print('⚠️ PyTorch not available')

try:
    import numba
    print('✅ Numba working')
except ImportError:
    print('⚠️ Numba not available')
"

echo "✅ Installation complete!"
echo "Run 'python3 step07_import_verification.py' to verify"
