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

1. In ChatGPT Codex, open **Project Settings → Setup**.
2. Set the **Setup Script** to `./setup_codex.sh` so Codex runs the helper automatically.
3. Set the **Working Directory** to `/workspace/Ares` so Codex executes the script from the project root.
4. Choose **Python 3.11** (or any interpreter compatible with the project's `^3.11` constraint, such as Python 3.12) as the runtime version.
5. In the **Environment Variables** section, add the following key/value pairs so every Codex-run command sees the installed packages:

   | Key | Value |
   | --- | ----- |
   | `PYTHONPATH` | `/workspace` |
   | `POETRY_VENV_IN_PROJECT` | `true` |

6. (Optional but recommended) If Codex allows specifying a **command prefix**, set it to `poetry run` so all execution happens inside the Poetry-managed virtual environment. Otherwise, remember to prefix any manual command yourself (e.g., `poetry run python src/launcher/ares_launcher.py ...`).
7. Save the configuration and start a new Codex session; the script will create `.venv` and install dependencies.
8. Codex automatically reuses the project virtual environment for subsequent commands. If you later open a local shell, you can still activate it manually (`source .venv/bin/activate`) or run tools with `poetry run ...`.

> **Quick reminder:** running the launcher with bare `python` (without the `poetry run` prefix or an activated `.venv`) will make the market-analysis modules appear missing because their dependencies are only installed inside the Poetry environment.

### Method 2: Using Requirements File

1. Run the export script: `python3 export_requirements.py`.
2. In Codex, specify `requirements.txt` as your dependencies file.
3. Set the **Working Directory** to `/workspace`.

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
