#!/usr/bin/env python3
"""
Migration script to convert from UV-based pyproject.toml to Poetry.
This script helps transition the existing dependency configuration.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def backup_existing_config():
    """Backup the existing pyproject.toml and poetry.lock."""
    project_root = Path(__file__).parent.parent
    
    # Backup existing files
    if (project_root / "pyproject.toml").exists():
        shutil.copy2(project_root / "pyproject.toml", project_root / "pyproject.toml.backup")
        logger.info("Backed up existing pyproject.toml")
    
    if (project_root / "poetry.lock").exists():
        shutil.copy2(project_root / "poetry.lock", project_root / "poetry.lock.backup")
        logger.info("Backed up existing poetry.lock")

def check_poetry_installation():
    """Check if Poetry is installed."""
    try:
        result = subprocess.run(['poetry', '--version'], 
                              capture_output=True, text=True, check=True)
        logger.info(f"Poetry found: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("Poetry not found. Please install Poetry first.")
        logger.info("Install Poetry with: curl -sSL https://install.python-poetry.org | python3 -")
        return False

def install_poetry_dependencies():
    """Install dependencies using the new Poetry configuration."""
    project_root = Path(__file__).parent.parent
    
    try:
        logger.info("Installing dependencies with Poetry...")
        subprocess.run(['poetry', 'install'], cwd=project_root, check=True)
        logger.info("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

def verify_installation():
    """Verify that key dependencies are working."""
    logger.info("Verifying installation...")
    
    try:
        # Test core dependencies
        import numpy as np
        import pandas as pd
        import scipy
        import sklearn
        logger.info("✅ Core scientific libraries working")
        
        # Test ML libraries
        import torch
        import tensorflow as tf
        logger.info("✅ Machine learning libraries working")
        
        # Test financial libraries
        import vectorbt as vbt
        import ccxt
        logger.info("✅ Financial analysis libraries working")
        
        # Test clustering libraries
        import hdbscan
        import optuna
        logger.info("✅ Clustering and optimization libraries working")
        
        logger.info("🎉 All dependencies verified successfully!")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False

def cleanup_old_files():
    """Clean up old dependency files if migration is successful."""
    project_root = Path(__file__).parent.parent
    
    # Files to potentially clean up
    old_files = [
        "pyproject.toml.backup",
        "poetry.lock.backup"
    ]
    
    for file in old_files:
        file_path = project_root / file
        if file_path.exists():
            response = input(f"Remove backup file {file}? (y/N): ")
            if response.lower() == 'y':
                file_path.unlink()
                logger.info(f"Removed {file}")

def main():
    """Main migration process."""
    logger.info("🚀 Starting migration to Poetry...")
    
    # Step 1: Backup existing configuration
    logger.info("Step 1: Backing up existing configuration...")
    backup_existing_config()
    
    # Step 2: Check Poetry installation
    logger.info("Step 2: Checking Poetry installation...")
    if not check_poetry_installation():
        sys.exit(1)
    
    # Step 3: Install dependencies
    logger.info("Step 3: Installing dependencies...")
    if not install_poetry_dependencies():
        logger.error("Failed to install dependencies. Check the error messages above.")
        sys.exit(1)
    
    # Step 4: Verify installation
    logger.info("Step 4: Verifying installation...")
    if not verify_installation():
        logger.error("Verification failed. Some dependencies may not be working correctly.")
        sys.exit(1)
    
    # Step 5: Cleanup (optional)
    logger.info("Step 5: Cleanup...")
    cleanup_old_files()
    
    logger.info("🎉 Migration completed successfully!")
    logger.info("You can now use Poetry commands:")
    logger.info("  poetry shell          # Activate virtual environment")
    logger.info("  poetry run python ares_launcher.py step01  # Run the application")
    logger.info("  poetry show           # Show installed packages")

if __name__ == "__main__":
    main()
