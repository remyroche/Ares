#!/usr/bin/env python3
"""
Environment Setup Script for TAS

This script sets up the development environment and installs dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_system_dependencies():
    """Install system-level dependencies."""
    commands = [
        ("apt update", "Updating package list"),
        ("apt install -y python3-venv python3-dev build-essential", "Installing system dependencies"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    return True

def create_virtual_environment():
    """Create a virtual environment."""
    venv_path = Path("/workspace/venv")
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    return run_command("python3 -m venv /workspace/venv", "Creating virtual environment")

def install_python_dependencies():
    """Install Python dependencies."""
    venv_pip = "/workspace/venv/bin/pip"
    
    # Upgrade pip first
    if not run_command(f"{venv_pip} install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install core dependencies
    core_deps = [
        "numpy>=1.21.0",
        "pandas>=1.3.0", 
        "scikit-learn>=1.0.0",
        "scipy>=1.9.0"
    ]
    
    for dep in core_deps:
        if not run_command(f"{venv_pip} install {dep}", f"Installing {dep}"):
            return False
    
    # Install ML dependencies
    ml_deps = [
        "xgboost>=1.5.0",
        "lightgbm>=3.2.0",
        "hmmlearn>=0.2.7",
        "optuna>=2.10.0",
        "joblib>=1.1.0"
    ]
    
    for dep in ml_deps:
        if not run_command(f"{venv_pip} install {dep}", f"Installing {dep}"):
            return False
    
    return True

def test_imports():
    """Test that core imports work."""
    venv_python = "/workspace/venv/bin/python"
    
    test_script = """
import sys
sys.path.append('/workspace')

try:
    import numpy as np
    print("✅ NumPy imported successfully")
except ImportError as e:
    print(f"❌ NumPy import failed: {e}")
    sys.exit(1)

try:
    import pandas as pd
    print("✅ Pandas imported successfully")
except ImportError as e:
    print(f"❌ Pandas import failed: {e}")
    sys.exit(1)

try:
    import sklearn
    print("✅ Scikit-learn imported successfully")
except ImportError as e:
    print(f"❌ Scikit-learn import failed: {e}")
    sys.exit(1)

print("✅ All core dependencies imported successfully")
"""
    
    return run_command(f"{venv_python} -c '{test_script}'", "Testing core imports")

def main():
    """Main setup function."""
    print("🚀 Setting up TAS development environment...")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install system dependencies
    if not install_system_dependencies():
        print("⚠️  System dependency installation failed, continuing...")
    
    # Create virtual environment
    if not create_virtual_environment():
        sys.exit(1)
    
    # Install Python dependencies
    if not install_python_dependencies():
        sys.exit(1)
    
    # Test imports
    if not test_imports():
        sys.exit(1)
    
    print("\n🎉 Environment setup completed successfully!")
    print("\nTo activate the virtual environment, run:")
    print("source /workspace/venv/bin/activate")
    print("\nTo test the TAS system, run:")
    print("/workspace/venv/bin/python -c \"import sys; sys.path.append('/workspace'); import src.analyst.analyst; print('TAS imports working!')\"")

if __name__ == "__main__":
    main()