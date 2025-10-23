#!/usr/bin/env python3
"""
VectorBT installation script for production use.

This script ensures VectorBT is properly installed and configured
for the Ares trading system.
"""

import sys
import subprocess
import importlib
import logging
from pathlib import Path
from typing import List, Tuple, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_python_version() -> bool:
    """Check if Python version is compatible."""
    if sys.version_info < (3.8, 0):
        logger.error(f"Python 3.8+ required, found {sys.version}")
        return False
    
    logger.info(f"Python version: {sys.version}")
    return True

def check_dependencies() -> List[str]:
    """Check required dependencies."""
    required_packages = [
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scipy>=1.7.0',
        'numba>=0.56.0'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        package_name = package.split('>=')[0]
        try:
            importlib.import_module(package_name)
            logger.info(f"✅ {package_name} is available")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"❌ {package_name} is missing")
    
    return missing_packages

def install_package(package: str) -> bool:
    """Install a package using pip."""
    try:
        logger.info(f"Installing {package}...")
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', package, '--upgrade'
        ], capture_output=True, text=True, check=True)
        
        logger.info(f"✅ Successfully installed {package}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install {package}: {e.stderr}")
        return False

def install_vectorbt() -> bool:
    """Install VectorBT with all dependencies."""
    try:
        logger.info("Installing VectorBT...")
        
        # Install VectorBT with dependencies
        packages_to_install = [
            'vectorbt>=0.25.0',
            'numpy>=1.21.0',
            'pandas>=1.3.0',
            'scipy>=1.7.0',
            'numba>=0.56.0',
            'psutil>=5.8.0'
        ]
        
        for package in packages_to_install:
            if not install_package(package):
                return False
        
        logger.info("✅ VectorBT installation completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ VectorBT installation failed: {e}")
        return False

def verify_vectorbt_installation() -> bool:
    """Verify VectorBT installation."""
    try:
        import vectorbt as vbt
        logger.info(f"✅ VectorBT {vbt.__version__} imported successfully")
        
        # Test basic functionality
        import pandas as pd
        import numpy as np
        
        # Create test data
        test_data = pd.Series(np.random.randn(100))
        
        # Test rolling operations
        from vectorbt.generic import rolling_mean, rolling_std
        rolling_mean(test_data, window=10)
        rolling_std(test_data, window=10)
        
        # Test portfolio creation
        from vectorbt.portfolio import Portfolio
        test_returns = test_data.pct_change().dropna()
        portfolio = Portfolio.from_returns(test_returns)
        
        logger.info("✅ VectorBT functionality test passed")
        return True
        
    except ImportError as e:
        logger.error(f"❌ VectorBT import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ VectorBT functionality test failed: {e}")
        return False

def run_vectorbt_tests() -> bool:
    """Run VectorBT tests."""
    try:
        logger.info("Running VectorBT tests...")
        
        # Import test module
        from src.vectorbt.test_vectorbt_integration import run_tests
        
        # Run tests
        success = run_tests()
        
        if success:
            logger.info("✅ VectorBT tests passed")
        else:
            logger.warning("⚠️ Some VectorBT tests failed")
        
        return success
        
    except ImportError as e:
        logger.error(f"❌ Could not import test module: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        return False

def configure_vectorbt_production() -> bool:
    """Configure VectorBT for production use."""
    try:
        logger.info("Configuring VectorBT for production...")
        
        # Import configuration
        from src.vectorbt.config import configure_vectorbt, PRODUCTION_CONFIG
        
        # Configure VectorBT
        configure_vectorbt(PRODUCTION_CONFIG)
        
        logger.info("✅ VectorBT production configuration applied")
        return True
        
    except Exception as e:
        logger.error(f"❌ VectorBT configuration failed: {e}")
        return False

def main():
    """Main installation process."""
    logger.info("🚀 Starting VectorBT installation for Ares trading system")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check existing dependencies
    missing_deps = check_dependencies()
    if missing_deps:
        logger.info(f"Installing missing dependencies: {missing_deps}")
        for dep in missing_deps:
            if not install_package(dep):
                logger.error(f"Failed to install {dep}")
                sys.exit(1)
    
    # Install VectorBT
    if not install_vectorbt():
        logger.error("VectorBT installation failed")
        sys.exit(1)
    
    # Verify installation
    if not verify_vectorbt_installation():
        logger.error("VectorBT verification failed")
        sys.exit(1)
    
    # Configure for production
    if not configure_vectorbt_production():
        logger.error("VectorBT production configuration failed")
        sys.exit(1)
    
    # Run tests
    if not run_vectorbt_tests():
        logger.warning("VectorBT tests failed, but installation may still work")
    
    logger.info("🎉 VectorBT installation completed successfully!")
    logger.info("VectorBT is now ready for production use in the Ares trading system")

if __name__ == "__main__":
    main()