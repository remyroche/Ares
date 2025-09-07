"""
Setup script for Enhanced Step 1.5 Data Converter Validator.

This script provides installation and setup functionality for the enhanced
step01_5 data converter validator with comprehensive function call monitoring.
"""
import sys
import subprocess
import importlib
from pathlib import Path

def check_python_version() -> bool:
    """Check if Python version meets requirements."""
    if sys.version_info < (3, 8):
        print('❌ Error: Python 3.8 or higher is required')
        print(f'   Current version: {sys.version}')
        return False
    print(f'✅ Python version check passed: {sys.version}')
    return True

def check_dependencies() -> bool:
    """Check if required dependencies are installed."""
    required_deps = {'pandas': '1.3.0', 'psutil': '5.8.0'}
    optional_deps = {'numpy': '1.20.0', 'pyarrow': '5.0.0', 'fastparquet': '0.7.0'}
    print('\n🔍 Checking dependencies...')
    missing_required = []
    for dep, min_version in required_deps.items():
        try:
            module = importlib.import_module(dep)
            version = getattr(module, '__version__', 'unknown')
            print(f'✅ {dep}: {version}')
        except ImportError:
            missing_required.append(dep)
            print(f'❌ {dep}: Not installed')
    missing_optional = []
    for dep, min_version in optional_deps.items():
        try:
            module = importlib.import_module(dep)
            version = getattr(module, '__version__', 'unknown')
            print(f'✅ {dep}: {version} (optional)')
        except ImportError:
            missing_optional.append(dep)
            print(f'⚠️ {dep}: Not installed (optional)')
    if missing_required:
        print(f"\n❌ Missing required dependencies: {', '.join(missing_required)}")
        return False
    if missing_optional:
        print(f"\n⚠️ Missing optional dependencies: {', '.join(missing_optional)}")
        print('   These are recommended for enhanced functionality')
    return True

def install_dependencies() -> bool:
    """Install required dependencies."""
    print('\n📦 Installing dependencies...')
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pandas >= 1.3.0', 'psutil >= 5.8.0'])
        print('✅ Core dependencies installed successfully')
        install_optional = input('\n❓ Install optional dependencies for enhanced functionality? (y/n): ').lower().strip()
        if install_optional in ['y', 'yes']:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyarrow >= 5.0.0', 'fastparquet >= 0.7.0'])
            print('✅ Optional dependencies installed successfully')
        return True
    except subprocess.CalledProcessError as e:
        print(f'❌ Error installing dependencies: {e}')
        return False

def verify_installation() -> bool:
    """Verify that the enhanced validator can be imported and used."""
    print('\n🔍 Verifying installation...')
    try:
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))
        from step01_5_data_converter_validator import Step1_5DataConverterValidator, FunctionCallMonitor, HealthCheckSystem, run_validator
        print('✅ Enhanced validator imports successful')
        import asyncio

        async def test_basic_functionality() -> bool:
            """Test basic functionality of the enhanced validator."""
            try:
                config = {}
                validator = Step1_5DataConverterValidator(config)
                health_checker = HealthCheckSystem(validator.logger)
                health_results = await health_checker.run_comprehensive_health_check()
                print(f"✅ Health check system working: {health_results['overall_status']}")
                print(f'✅ Function call monitor initialized')
                print(f'✅ Enhanced validator ready for use')
                return True
            except Exception as e:
                print(f'❌ Error testing functionality: {e}')
                return False
        success = asyncio.run(test_basic_functionality())
        return success
    except ImportError as e:
        print(f'❌ Import error: {e}')
        return False
    except Exception as e:
        print(f'❌ Verification error: {e}')
        return False

def main() -> None:
    """Main setup function."""
    print('🚀 Enhanced Step 1.5 Data Converter Validator Setup')
    print('=' * 60)
    if not check_python_version():
        sys.exit(1)
    deps_ok = check_dependencies()
    if not deps_ok:
        install_choice = input('\n❓ Install missing dependencies? (y/n): ').lower().strip()
        if install_choice in ['y', 'yes']:
            if not install_dependencies():
                sys.exit(1)
        else:
            print('❌ Cannot proceed without required dependencies')
            sys.exit(1)
    if not verify_installation():
        print('❌ Installation verification failed')
        sys.exit(1)
    print('\n🎉 Enhanced Step 1.5 Data Converter Validator setup completed successfully!')
    print('\n📋 Next steps:')
    print('   1. Run the enhanced validator: python step01_5_data_converter_validator.py')
    print('   2. Check the comprehensive test suite in the main function')
    print('   3. Review the detailed documentation in the module docstring')
    print('\n📚 For more information, see the module documentation and requirements file.')
if __name__ == '__main__':
    main()