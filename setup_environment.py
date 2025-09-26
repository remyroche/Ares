#!/usr/bin/env python3
"""Environment setup script that bootstraps Poetry and installs dependencies.

This script relies on the ``poetry.lock`` file for dependency resolution so that
local environments match the versions tracked in version control.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from textwrap import dedent


PROJECT_ROOT = Path(__file__).resolve().parent
POETRY_COMMAND: str | None = None


def _poetry_command_available(command: str) -> bool:
    """Return True if ``command`` can successfully invoke Poetry."""

    try:
        subprocess.run(
            f"{command} --version",
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return False

    return True


def run_command(command: str, description: str) -> bool:
    """Run ``command`` in the shell, printing a friendly status message."""

    print(f"🔄 {description}...")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            print(result.stdout.strip())
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as exc:  # pragma: no cover - CLI helper
        print(f"❌ {description} failed: {exc}")
        if exc.stdout:
            print(exc.stdout)
        if exc.stderr:
            print(exc.stderr)
        return False


def check_python_version() -> bool:
    """Ensure the interpreter satisfies the minimum Poetry requirement (3.8+)."""

    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False

    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def ensure_poetry_installed() -> bool:
    """Install Poetry if necessary and verify it works."""

    global POETRY_COMMAND

    python_module_cmd = "python3 -m poetry"
    if _poetry_command_available(python_module_cmd):
        POETRY_COMMAND = python_module_cmd
        print("✅ Poetry is available via 'python3 -m poetry'")
        return True

    poetry_executable = shutil.which("poetry")
    if poetry_executable and _poetry_command_available("poetry"):
        POETRY_COMMAND = "poetry"
        print(f"✅ Found Poetry at {poetry_executable}")
        return True

    print("ℹ️ Poetry was not detected on PATH; attempting installation via pip...")
    if not run_command(
        "python3 -m pip install --user --upgrade poetry",
        "Installing Poetry",
    ):
        return False

    if _poetry_command_available(python_module_cmd):
        POETRY_COMMAND = python_module_cmd
        print("✅ Poetry installed successfully via 'python3 -m poetry'")
        return True

    poetry_executable = shutil.which("poetry")
    if poetry_executable and _poetry_command_available("poetry"):
        POETRY_COMMAND = "poetry"
        print(f"✅ Poetry installed successfully at {poetry_executable}")
        return True

    print("❌ Failed to verify Poetry installation")
    return False


def install_dependencies_with_poetry() -> bool:
    """Use Poetry to create a virtual environment from ``poetry.lock``."""

    lock_file = PROJECT_ROOT / "poetry.lock"
    if not lock_file.exists():
        print("❌ poetry.lock is missing; cannot perform a reproducible install")
        return False

    if POETRY_COMMAND is None:
        raise RuntimeError("Poetry must be installed before installing dependencies")

    description = "Installing dependencies from poetry.lock"
    command = f"cd {PROJECT_ROOT} && {POETRY_COMMAND} install --no-root --sync"
    return run_command(command, description)


def test_core_imports() -> bool:
    """Validate that critical dependencies import correctly through Poetry."""

    test_script = dedent(
        """
        from importlib import import_module

        modules = [
            ("numpy", "NumPy"),
            ("pandas", "Pandas"),
            ("sklearn", "Scikit-learn"),
        ]

        failures = []
        for module_name, label in modules:
            try:
                import_module(module_name)
                print(f"✅ {label} imported successfully")
            except ImportError as exc:
                failures.append(f"❌ {label} import failed: {exc}")

        if failures:
            for failure in failures:
                print(failure)
            raise SystemExit(1)

        print("✅ All core dependencies imported successfully")
        """
    ).strip().replace("\n", "\\n").replace('"', '\\"')

    if POETRY_COMMAND is None:
        raise RuntimeError("Poetry must be installed before verifying imports")

    command = (
        f"cd {PROJECT_ROOT} && {POETRY_COMMAND} run python -c \"{test_script}\""
    )
    return run_command(command, "Verifying core imports via Poetry")


def main() -> None:
    """Entry point for environment setup."""

    print("🚀 Setting up the TAS development environment using Poetry...")

    if not check_python_version():
        sys.exit(1)

    if not ensure_poetry_installed():
        sys.exit(1)

    if not install_dependencies_with_poetry():
        sys.exit(1)

    if not test_core_imports():
        sys.exit(1)

    print("\n🎉 Environment setup completed successfully!")
    print("\nNext steps:")
    print("  • Run 'poetry shell' to enter the virtual environment")
    print(
        "  • Or prefix commands with 'poetry run', e.g.\n"
        "    poetry run python path/to/script.py"
    )


if __name__ == "__main__":
    main()
