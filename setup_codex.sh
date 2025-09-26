#!/bin/bash

# ChatGPT Codex Setup Script for Poetry Dependencies
# This script ensures Codex installs the exact dependencies from poetry.lock

echo "🚀 Setting up environment for ChatGPT Codex..."

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "📦 Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi

# Verify Poetry installation
echo "✅ Poetry version: $(poetry --version)"

# Install dependencies from poetry.lock
echo "📦 Installing dependencies from poetry.lock..."
poetry install --no-dev

# Verify key dependencies are available
echo "🧪 Verifying key dependencies..."
python3 -c "
import sys
sys.path.append('/workspace')

# Test core imports
try:
    import numpy as np
    print('✅ NumPy:', np.__version__)
except ImportError as e:
    print(f'❌ NumPy import failed: {e}')

try:
    import pandas as pd
    print('✅ Pandas:', pd.__version__)
except ImportError as e:
    print(f'❌ Pandas import failed: {e}')

try:
    import sklearn
    print('✅ Scikit-learn:', sklearn.__version__)
except ImportError as e:
    print(f'❌ Scikit-learn import failed: {e}')

try:
    import optuna
    print('✅ Optuna:', optuna.__version__)
except ImportError as e:
    print(f'❌ Optuna import failed: {e}')

try:
    import xgboost as xgb
    print('✅ XGBoost:', xgb.__version__)
except ImportError as e:
    print(f'❌ XGBoost import failed: {e}')

try:
    import lightgbm as lgb
    print('✅ LightGBM:', lgb.__version__)
except ImportError as e:
    print(f'❌ LightGBM import failed: {e}')

print('🎉 Dependency verification completed')
"

echo "✅ Codex environment setup completed successfully!"