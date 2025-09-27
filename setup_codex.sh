#!/bin/bash

# ChatGPT Codex Setup Script for Poetry Dependencies
# This script ensures Codex installs the exact dependencies from poetry.lock

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${ROOT_DIR}/.venv"

ensure_poetry() {
    if command -v poetry >/dev/null 2>&1; then
        return
    fi

    echo "📦 Poetry not found. Installing Poetry..."

    # Prefer installing via pip because Codex environments already have Python available.
    if python3 -m pip install --user --upgrade poetry >/dev/null 2>&1; then
        export PATH="${HOME}/.local/bin:${PATH}"
    else
        echo "⚠️ pip installation failed. Using the official Poetry installer..."
        curl -sSL https://install.python-poetry.org | python3 - --yes >/dev/null
        export PATH="${HOME}/.local/bin:${PATH}"
    fi

    if ! command -v poetry >/dev/null 2>&1; then
        echo "❌ Unable to install Poetry. Aborting." >&2
        exit 1
    fi
}

cleanup() {
    # Remove the temporary requirements file if it exists from older runs.
    rm -f "${ROOT_DIR}/.codex-requirements.txt"
}
trap cleanup EXIT

echo "🚀 Setting up environment for ChatGPT Codex..."

# Ensure we have an up-to-date pip that can handle modern wheels
python3 -m pip install --upgrade pip >/dev/null 2>&1 || true

ensure_poetry

# Ensure Poetry keeps the virtual environment inside the project for Codex
export POETRY_VENV_IN_PROJECT="${POETRY_VENV_IN_PROJECT:-true}"
export PYTHONPATH="${PYTHONPATH:-${ROOT_DIR}}"

# Use the project's preferred Python if available
if [[ -x "${VENV_PATH}/bin/python" ]]; then
    echo "🔁 Re-using existing Poetry virtual environment..."
else
    echo "🛠️ Configuring Poetry environment..."
    poetry env use "$(command -v python3)" >/dev/null 2>&1 || true
fi

# Install the dependencies specified in poetry.lock without development packages
echo "📦 Installing project dependencies via Poetry..."
poetry install --no-interaction --no-root --no-dev

echo "🧪 Verifying key dependencies inside the Poetry environment..."
poetry run python - <<'PY'
import importlib

deps = ["numpy", "pandas", "sklearn", "optuna", "xgboost", "lightgbm"]

missing = []
for module_name in deps:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        print(f"✅ {module_name}: {version}")
    except Exception as exc:  # pragma: no cover - diagnostic output for Codex
        print(f"❌ {module_name} import failed: {exc}")
        missing.append(module_name)

if missing:
    raise SystemExit(f"Missing required dependencies: {', '.join(missing)}")

print("🎉 Dependency verification completed")
PY

echo "✅ Codex environment setup completed successfully!"

if [[ -f "${VENV_PATH}/bin/activate" ]]; then
    cat <<'EOM'

Next steps:
  • Codex already uses this virtual environment for subsequent commands.
  • If you open another shell locally, activate it with:
      source .venv/bin/activate
  • Or run project tools directly with:
      poetry run python your_script.py
  • Deactivate the manual shell session with:
      deactivate

EOM
fi
