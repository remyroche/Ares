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

cd "${ROOT_DIR}"

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

# Poetry 1.7 deprecated --no-dev in favor of --without dev. Detect which flag
# is supported so the script remains compatible with a wider range of Poetry
# versions that may be preinstalled in the execution environment.
INSTALL_ARGS=("--no-interaction" "--no-root")

# Only attempt to exclude development dependencies if the project actually
# defines any. Busybox grep treats patterns beginning with ``-`` as options, so
# pass ``--`` to ensure they are interpreted literally.
if grep -Eq '^\[tool\.poetry\.(dev-dependencies|group\.dev)]' "${ROOT_DIR}/pyproject.toml"; then
    if poetry help install 2>/dev/null | grep -q -- "--without"; then
        INSTALL_ARGS+=("--without" "dev")
    elif poetry help install 2>/dev/null | grep -q -- "--no-dev"; then
        INSTALL_ARGS+=("--no-dev")
    fi
fi

poetry install "${INSTALL_ARGS[@]}"

if [[ -f "${ROOT_DIR}/requirements.txt" ]]; then
    echo "📚 Installing supplemental requirements from requirements.txt via pip..."
    if ! poetry run python - <<'PY'
import subprocess
import sys
from pathlib import Path

req_path = Path("requirements.txt")
if req_path.exists():
    cmd = [sys.executable, "-m", "pip", "install", "-r", str(req_path)]
    print("   →", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
PY
    then
        echo "❌ Failed to install supplemental pip requirements." >&2
        exit 1
    fi
fi

echo "🧪 Verifying key dependencies inside the Poetry environment..."
poetry run python - <<'PY'
import importlib

deps = [
    "numpy",
    "pandas",
    "sklearn",
    "optuna",
    "xgboost",
    "lightgbm",
    "torch",
    "tensorflow",
    "transformers",
    "matplotlib",
    "seaborn",
    "networkx",
    "plotly",
    "graphviz",
    "squarify",
    "rich",
    "ccxt",
    "yfinance",
    "talib",
    "asyncio_mqtt",
    "aiohttp",
    "websockets",
    "yaml",
    "structlog",
    "loguru",
]

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
