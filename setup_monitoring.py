#!/usr/bin/env python3
"""
Setup script for Comprehensive Monitoring System

This script sets up the monitoring system dependencies and validates the installation.
"""

import sys
import subprocess
import importlib
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} is compatible")
    return True


def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False


def check_package(package_name, import_name=None):
    """Check if a package is installed and can be imported."""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} is available")
        return True
    except ImportError:
        print(f"❌ {package_name} is not available")
        return False


def install_core_dependencies():
    """Install core dependencies for the monitoring system."""
    print("📦 Installing core dependencies...")
    
    core_packages = [
        "psutil>=5.8.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0"
    ]
    
    for package in core_packages:
        print(f"Installing {package}...")
        if install_package(package):
            print(f"✅ {package} installed successfully")
        else:
            print(f"❌ Failed to install {package}")
            return False
    
    return True


def install_optional_dependencies():
    """Install optional dependencies for enhanced functionality."""
    print("📦 Installing optional dependencies...")
    
    optional_packages = [
        "structlog>=21.0.0",
        "prometheus-client>=0.12.0",
        "pytest>=6.0.0",
        "pytest-asyncio>=0.18.0",
        "black>=22.0.0",
        "flake8>=4.0.0",
        "mypy>=0.950"
    ]
    
    for package in optional_packages:
        print(f"Installing {package}...")
        if install_package(package):
            print(f"✅ {package} installed successfully")
        else:
            print(f"⚠️ Failed to install {package} (optional)")


def validate_installation():
    """Validate that all required packages are installed."""
    print("🔍 Validating installation...")
    
    required_packages = [
        ("psutil", "psutil"),
        ("pandas", "pandas"),
        ("numpy", "numpy")
    ]
    
    all_installed = True
    for package_name, import_name in required_packages:
        if not check_package(package_name, import_name):
            all_installed = False
    
    return all_installed


def test_monitoring_system():
    """Test the monitoring system components."""
    print("🧪 Testing monitoring system components...")
    
    try:
        # Test function call monitor
        from src.utils.function_call_monitor import get_function_call_monitor
        monitor = get_function_call_monitor()
        print("✅ Function call monitor is working")
        
        # Test validation framework
        from src.utils.function_validation_framework import get_function_validator
        validator = get_function_validator()
        print("✅ Function validation framework is working")
        
        # Test error handler
        from src.utils.enhanced_error_handler import get_error_handler
        error_handler = get_error_handler()
        print("✅ Enhanced error handler is working")
        
        # Test comprehensive monitoring
        from src.training.steps.data_collection.step01_comprehensive_monitoring import Step01ComprehensiveMonitoring
        print("✅ Comprehensive monitoring system is working")
        
        return True
        
    except Exception as e:
        print(f"❌ Monitoring system test failed: {e}")
        return False


def create_monitoring_config():
    """Create a default monitoring configuration file."""
    config_content = """# Comprehensive Monitoring System Configuration

monitoring:
  enabled: true
  level: comprehensive  # basic, standard, comprehensive
  export_reports: true
  reports_directory: monitoring_reports
  
  function_calls:
    track_performance: true
    track_memory: true
    track_cpu: true
    export_detailed_reports: true
    
  validation:
    check_parameters: true
    check_security: true
    check_business_logic: true
    validation_level: standard  # basic, standard, comprehensive
    
  error_handling:
    enable_recovery: true
    track_errors: true
    export_error_reports: true
    recovery_strategies:
      retry: true
      fallback: true
      skip: true
      
  performance:
    track_timing: true
    track_memory: true
    track_cpu: true
    optimization_recommendations: true
    
  logging:
    structured_logging: true
    real_time_monitoring: true
    log_level: INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    export_logs: true
"""
    
    config_path = Path("monitoring_config.yaml")
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Monitoring configuration created: {config_path}")


def main():
    """Main setup function."""
    print("🚀 Setting up Comprehensive Monitoring System")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install core dependencies
    if not install_core_dependencies():
        print("❌ Failed to install core dependencies")
        sys.exit(1)
    
    # Install optional dependencies
    install_optional_dependencies()
    
    # Validate installation
    if not validate_installation():
        print("❌ Installation validation failed")
        sys.exit(1)
    
    # Test monitoring system
    if not test_monitoring_system():
        print("❌ Monitoring system test failed")
        sys.exit(1)
    
    # Create configuration file
    create_monitoring_config()
    
    print("=" * 60)
    print("🎉 Comprehensive Monitoring System setup completed successfully!")
    print("=" * 60)
    print("📋 Next steps:")
    print("1. Review the monitoring configuration in monitoring_config.yaml")
    print("2. Run the test suite: python test_comprehensive_step01_monitoring.py")
    print("3. Start using the monitoring system in your code")
    print("4. Check the documentation in STEP01_COMPREHENSIVE_MONITORING_README.md")
    print("=" * 60)


if __name__ == "__main__":
    main()