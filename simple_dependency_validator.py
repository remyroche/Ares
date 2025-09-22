#!/usr/bin/env python3
"""
Simple Dependency Validator

This script validates dependencies without requiring external packages,
providing detailed information about what's missing and how to fix it.
"""

import sys
import importlib
import subprocess
from pathlib import Path
from datetime import datetime

def print_banner():
    """Print validation banner."""
    print("=" * 80)
    print("🔍 TRAINING PIPELINE DEPENDENCY VALIDATION")
    print("=" * 80)
    print(f"⏰ Started at: {datetime.now()}")
    print("")

def check_dependency(package_name, import_name=None, required=True):
    """Check if a dependency is available."""
    import_name = import_name or package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'Unknown')
        print(f"  ✅ {package_name}: {version}")
        return True
    except ImportError as e:
        status = "❌ CRITICAL" if required else "⚠️  OPTIONAL"
        print(f"  {status} {package_name}: Missing ({e})")
        return False

def check_data_files():
    """Check if data files exist."""
    print("📂 Checking data files...")
    
    data_dir = Path("historical_data")
    if not data_dir.exists():
        print(f"  ❌ Data directory missing: {data_dir}")
        print(f"     💡 Create directory: mkdir -p {data_dir}")
        return False
    
    # Check for common data patterns
    patterns = [
        "binance/ETHUSDT/15m/*.parquet",
        "binance/ethusdt/15m/*.parquet", 
        "ETHUSDT_15m_*.parquet"
    ]
    
    found_files = []
    for pattern in patterns:
        files = list(data_dir.glob(pattern))
        found_files.extend(files)
    
    if found_files:
        print(f"  ✅ Found {len(found_files)} data files")
        for file_path in found_files[:3]:  # Show first 3
            print(f"     📄 {file_path}")
        if len(found_files) > 3:
            print(f"     📄 ... and {len(found_files) - 3} more files")
        return True
    else:
        print(f"  ❌ No data files found")
        print(f"     💡 Run data collection pipeline first")
        return False

def check_source_files():
    """Check if source files exist."""
    print("📋 Checking source files...")
    
    critical_files = [
        "src/training/steps/model_training/sub_pipeline.py",
        "src/training/utils/debug_utilities.py",
        "src/utils/tprint.py",
        "src/utils/logger.py"
    ]
    
    missing_files = []
    for file_path in critical_files:
        path = Path(file_path)
        if path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def generate_installation_commands():
    """Generate installation commands for missing dependencies."""
    print("\n" + "=" * 80)
    print("💡 INSTALLATION RECOMMENDATIONS")
    print("=" * 80)
    
    print("\n1️⃣ Try installing with --break-system-packages (if you have permission):")
    print("   pip3 install --break-system-packages numpy pandas psutil scikit-learn xgboost lightgbm")
    
    print("\n2️⃣ Or install system packages (if you have sudo access):")
    print("   sudo apt update")
    print("   sudo apt install python3-numpy python3-pandas python3-psutil python3-sklearn")
    
    print("\n3️⃣ Or use pipx for isolated installation:")
    print("   pipx install numpy pandas psutil scikit-learn")
    
    print("\n4️⃣ Create requirements.txt for future reference:")
    requirements = [
        "numpy>=1.21.0",
        "pandas>=1.3.0", 
        "psutil>=5.8.0",
        "scikit-learn>=1.0.0",
        "xgboost>=1.5.0",
        "lightgbm>=3.2.0",
        "hmmlearn>=0.2.7",
        "optuna>=2.10.0",
        "joblib>=1.1.0"
    ]
    
    with open("requirements.txt", "w") as f:
        for req in requirements:
            f.write(f"{req}\n")
    
    print("   ✅ Created requirements.txt")
    print("   📝 Install with: pip3 install -r requirements.txt")

def check_python_version():
    """Check Python version compatibility."""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    print(f"  📊 Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  ❌ Python 3.8+ required")
        return False
    else:
        print("  ✅ Python version compatible")
        return True

def main():
    """Main validation function."""
    print_banner()
    
    results = {}
    
    # Check Python version
    results['python_version'] = check_python_version()
    print()
    
    # Check critical dependencies
    print("📦 Checking critical dependencies...")
    critical_deps = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("psutil", "psutil"),
        ("scikit-learn", "sklearn"),
    ]
    
    critical_available = 0
    for package, import_name in critical_deps:
        if check_dependency(package, import_name, required=True):
            critical_available += 1
    
    results['critical_dependencies'] = critical_available == len(critical_deps)
    print()
    
    # Check optional dependencies
    print("📦 Checking optional dependencies...")
    optional_deps = [
        ("xgboost", "xgboost"),
        ("lightgbm", "lightgbm"), 
        ("hmmlearn", "hmmlearn"),
        ("optuna", "optuna"),
        ("joblib", "joblib"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn")
    ]
    
    optional_available = 0
    for package, import_name in optional_deps:
        if check_dependency(package, import_name, required=False):
            optional_available += 1
    
    results['optional_dependencies'] = optional_available
    print()
    
    # Check source files
    results['source_files'] = check_source_files()
    print()
    
    # Check data files
    results['data_files'] = check_data_files()
    print()
    
    # Summary
    print("=" * 80)
    print("📊 VALIDATION SUMMARY")
    print("=" * 80)
    
    total_score = 0
    max_score = 0
    
    # Python version (critical)
    if results['python_version']:
        print("  ✅ Python Version: Compatible")
        total_score += 2
    else:
        print("  ❌ Python Version: Incompatible")
    max_score += 2
    
    # Critical dependencies
    if results['critical_dependencies']:
        print("  ✅ Critical Dependencies: All available")
        total_score += 3
    else:
        print(f"  ❌ Critical Dependencies: {critical_available}/{len(critical_deps)} available")
    max_score += 3
    
    # Optional dependencies
    print(f"  📊 Optional Dependencies: {optional_available}/{len(optional_deps)} available")
    total_score += min(optional_available, 2)  # Max 2 points for optional
    max_score += 2
    
    # Source files
    if results['source_files']:
        print("  ✅ Source Files: All present")
        total_score += 2
    else:
        print("  ❌ Source Files: Some missing")
    max_score += 2
    
    # Data files
    if results['data_files']:
        print("  ✅ Data Files: Available")
        total_score += 1
    else:
        print("  ❌ Data Files: Missing")
    max_score += 1
    
    print(f"\n🎯 Overall Score: {total_score}/{max_score} ({total_score/max_score*100:.0f}%)")
    
    if total_score >= max_score * 0.8:
        print("🎉 READY: Training pipeline should work with minor issues")
    elif total_score >= max_score * 0.6:
        print("⚠️  PARTIAL: Training pipeline may work with some limitations")
    else:
        print("❌ NOT READY: Critical dependencies missing")
    
    # Generate installation recommendations
    if not results['critical_dependencies'] or optional_available < len(optional_deps) // 2:
        generate_installation_commands()
    
    print("\n" + "=" * 80)
    return total_score >= max_score * 0.6

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Validation failed: {str(e)}")
        sys.exit(1)