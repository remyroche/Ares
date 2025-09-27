#!/bin/bash

# ChatGPT Codex Setup Script for Poetry Dependencies
# This script ensures Codex installs the exact dependencies from poetry.lock

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="${ROOT_DIR}/.codex-requirements.txt"

cleanup() {
    rm -f "$REQ_FILE"
}
trap cleanup EXIT

echo "🚀 Setting up environment for ChatGPT Codex..."

# Ensure we have an up-to-date pip that can handle modern wheels
python3 -m pip install --upgrade pip >/dev/null 2>&1 || true

# If poetry is available, try exporting the main dependencies to a requirements file.
if command -v poetry &> /dev/null; then
    echo "📦 Exporting dependencies from poetry.lock via Poetry..."
    if ! poetry export --without-hashes --format requirements.txt --output "$REQ_FILE"; then
        echo "⚠️ Poetry export failed. Falling back to pyproject.toml parsing."
    fi
else
    echo "📦 Poetry not found. Generating requirements from pyproject.toml..."
fi

if [[ ! -f "$REQ_FILE" || ! -s "$REQ_FILE" ]]; then
    python3 - <<'PY'
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
pyproject_path = project_root / "pyproject.toml"

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - safety for 3.10 runners
    import tomli as tomllib

config = tomllib.loads(pyproject_path.read_text())
deps = config.get("tool", {}).get("poetry", {}).get("dependencies", {})

# Remove the python version specifier and keep the rest
requirements = [
    f"{name}{'' if name == 'python' else f'=={spec}' if spec.replace('.', '').isdigit() else f' {spec}'}"
    for name, spec in deps.items()
    if name.lower() != "python"
]

if not requirements:
    sys.exit("No dependencies found in pyproject.toml")

req_path = project_root / ".codex-requirements.txt"
req_path.write_text("\n".join(requirements) + "\n")
PY
fi

# Install the requirements captured above.
if [[ -f "$REQ_FILE" ]]; then
    echo "📦 Installing project dependencies..."
    python3 -m pip install -r "$REQ_FILE"
else
    echo "❌ Failed to generate requirements file. Aborting." >&2
    exit 1
fi

echo "🧪 Verifying key dependencies..."
python3 - <<'PY'
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
