# ChatGPT Codex Setup Guide

This guide explains how to configure ChatGPT Codex to work with this Poetry-based project.

## Files Created for Codex Integration

### 1. `setup_codex.sh`
- **Purpose**: Main setup script that Codex will execute
- **Function**: Installs Poetry (if needed) and runs `poetry install`
- **Usage**: Configure this as your setup script in Codex

### 2. `.codex-config.json`
- **Purpose**: Configuration file for Codex environment
- **Function**: Specifies setup script, Python version, and environment variables
- **Usage**: Reference this file when configuring your Codex environment

### 3. `export_requirements.py`
- **Purpose**: Alternative method to export Poetry dependencies
- **Function**: Creates `requirements.txt` from `poetry.lock`
- **Usage**: Run this script to generate a traditional requirements file

## How to Configure Codex

### Method 1: Using Setup Script (Recommended)

1. In ChatGPT Codex, go to your project settings
2. Set the **Setup Script** to: `./setup_codex.sh`
3. Set the **Working Directory** to: `/workspace`
4. Set the **Python Version** to: `3.11`

### Method 2: Using Requirements File

1. Run the export script: `python3 export_requirements.py`
2. In Codex, specify `requirements.txt` as your dependencies file
3. Set the **Working Directory** to: `/workspace`

## Environment Variables

The following environment variables are configured:
- `PYTHONPATH=/workspace` - Ensures Python can find your modules
- `POETRY_VENV_IN_PROJECT=true` - Keeps Poetry virtual environment in project

## Dependencies

This project uses Poetry for dependency management with the following key packages:
- **Core ML**: numpy, pandas, scikit-learn, scipy
- **Advanced ML**: xgboost, lightgbm, hmmlearn
- **Optimization**: optuna, joblib
- **System**: psutil

## Verification

After setup, Codex will automatically verify that key dependencies are importable:
- NumPy
- Pandas  
- Scikit-learn
- Optuna
- XGBoost
- LightGBM

## Troubleshooting

### Poetry Not Found
If Poetry is not installed, the setup script will automatically install it.

### Import Errors
Check that the `PYTHONPATH` environment variable includes `/workspace`.

### Version Conflicts
The setup uses `poetry install --no-dev` to avoid development dependencies that might conflict.

## Notes

- The setup script installs dependencies without development packages to avoid conflicts
- All dependencies are locked to specific versions in `poetry.lock`
- The virtual environment is created in the project directory for consistency