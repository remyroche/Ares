#!/usr/bin/env python3
import numpy as np
import pandas as pd
import warnings

"""
Step07 Dependency Management and Import Fix Script

This script addresses the critical dependency and import issues identified in the Step07 audit.
It provides multiple solutions for different deployment scenarios.
"""

import sys
import subprocess
import os
from pathlib import Path

def check_python_environment():
    """Check the current Python environment and available tools."""
    print("🔍 Checking Python Environment")
    print("=" * 50)
    
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Platform: {sys.platform}")
    
    # Check for pip
    try:
        import pip
        print(f"✅ pip available: {pip.__version__}")
    except ImportError:
        print("❌ pip not available")
    
    # Check for venv
    try:
        import venv
        import logging
        import typing

        print("✅ venv module available")
    except ImportError:
        print("❌ venv module not available")
    
    # Check for conda
    try:
        result = subprocess.run(['conda', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ conda available: {result.stdout.strip()}")
        else:
            print("❌ conda not available")
    except FileNotFoundError:
        print("❌ conda not available")

def create_requirements_file():
    """Create a comprehensive requirements.txt file for Step07."""
    requirements_content = """# Step07 Enhanced Matrix Operations - Dependencies
# Core scientific computing
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0

# Machine learning
scikit-learn>=1.0.0
torch>=1.12.0
numba>=0.56.0

# System monitoring
psutil>=5.8.0

# Optional but recommended
lightgbm>=3.3.0
xgboost>=1.6.0

# Development and testing
pytest>=6.0.0
pytest-asyncio>=0.18.0

# Visualization (optional)
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0
"""
    
    with open('requirements_step07.txt', 'w') as f:
        f.write(requirements_content)
    
    print("✅ Created requirements_step07.txt")

def create_conda_environment_file():
    """Create a conda environment.yml file for Step07."""
    conda_content = """name: step07-matrix-ops
channels:
  - conda-forge
  - pytorch
  - nvidia
dependencies:
  - python=3.9
  - numpy>=1.21.0
  - pandas>=1.3.0
  - scipy>=1.7.0
  - scikit-learn>=1.0.0
  - pytorch>=1.12.0
  - numba>=0.56.0
  - psutil>=5.8.0
  - lightgbm>=3.3.0
  - xgboost>=1.6.0
  - pytest>=6.0.0
  - matplotlib>=3.5.0
  - seaborn>=0.11.0
  - plotly>=5.0.0
  - pip
  - pip:
    - pytest-asyncio>=0.18.0
"""
    
    with open('environment_step07.yml', 'w') as f:
        f.write(conda_content)
    
    print("✅ Created environment_step07.yml")

def create_dockerfile():
    """Create a Dockerfile for Step07 with all dependencies."""
    dockerfile_content = """FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements
COPY requirements_step07.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_step07.txt

# Copy source code
COPY . .

# Set environment variables
ENV PYTHONPATH=/workspace
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

# Create cache directory
RUN mkdir -p /tmp/numba_cache

# Default command
CMD ["python", "step07_import_verification.py"]
"""
    
    with open('Dockerfile.step07', 'w') as f:
        f.write(dockerfile_content)
    
    print("✅ Created Dockerfile.step07")

def create_import_fix_module():
    """Create a module to fix import issues in Step07."""
    import_fix_content = '''"""
Step07 Import Fix Module

This module provides safe imports with proper fallback handling
to resolve the import chain issues identified in the audit.
"""

import sys
from typing import Optional, Any, Dict
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

class SafeImporter:
    """Safe import utility with fallback handling."""
    
    def __init__(self):
        self.import_cache = {}
        self.fallback_modules = {}
    
    def safe_import(self, module_name: str, fallback: Any = None) -> Any:
        """Safely import a module with fallback."""
        if module_name in self.import_cache:
            return self.import_cache[module_name]
        
        try:
            module = __import__(module_name)
            self.import_cache[module_name] = module
            return module
        except ImportError as e:
            if fallback is not None:
                self.fallback_modules[module_name] = fallback
                return fallback
            else:
                warnings.warn(f"Failed to import {module_name}: {e}")
                return None
    
    def get_import_status(self) -> Dict[str, bool]:
        """Get status of all imports."""
        status = {}
        for module_name in self.import_cache:
            status[module_name] = True
        for module_name in self.fallback_modules:
            status[module_name] = False
        return status

# Global safe importer instance
safe_importer = SafeImporter()

# Core scientific computing imports
numpy = safe_importer.safe_import('numpy')
pandas = safe_importer.safe_import('pandas')
scipy = safe_importer.safe_import('scipy')
sklearn = safe_importer.safe_import('sklearn')
torch = safe_importer.safe_import('torch')
numba = safe_importer.safe_import('numba')
psutil = safe_importer.safe_import('psutil')

# Optional ML libraries
lightgbm = safe_importer.safe_import('lightgbm')
xgboost = safe_importer.safe_import('xgboost')

# Project-specific imports with fallbacks
def get_system_logger():
    """Get system logger with fallback."""
    try:
        from src.utils.logger import system_logger
        return system_logger
    except ImportError:
        import logging
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger('step07_fallback')

def get_handles_errors_decorator():
    """Get handles_errors decorator with fallback."""
    try:
        from src.core.decorators import handles_errors
        return handles_errors
    except ImportError:
        def fallback_decorator(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
        return fallback_decorator

def get_base_step():
    """Get BaseStep class with fallback."""
    try:
        from src.training.base_step import BaseStep
        return BaseStep
    except ImportError:
        class FallbackBaseStep:
            def __init__(self, config, step_id, step_name):
                self.config = config
                self.step_id = step_id
                self.step_name = step_name
        return FallbackBaseStep

# Initialize fallback components
system_logger = get_system_logger()
handles_errors = get_handles_errors_decorator()
BaseStep = get_base_step()

def check_dependencies():
    """Check if all required dependencies are available."""
    required_modules = ['numpy', 'pandas', 'sklearn', 'torch', 'numba', 'psutil']
    missing_modules = []
    
    for module_name in required_modules:
        if safe_importer.safe_import(module_name) is None:
            missing_modules.append(module_name)
    
    if missing_modules:
        print(f"❌ Missing required modules: {missing_modules}")
        return False
    else:
        print("✅ All required modules available")
        return True

def get_import_summary():
    """Get summary of import status."""
    status = safe_importer.get_import_status()
    print("📊 Import Status Summary:")
    for module, available in status.items():
        status_icon = "✅" if available else "❌"
        print(f"  {status_icon} {module}")
    return status

if __name__ == "__main__":
    print("🔍 Step07 Import Fix Module")
    print("=" * 40)
    check_dependencies()
    get_import_summary()
'''
    
    with open('src/utils/step07_import_fix.py', 'w') as f:
        f.write(import_fix_content)
    
    print("✅ Created src/utils/step07_import_fix.py")

def create_simplified_step07():
    """Create a simplified version of Step07 with fixed imports."""
    simplified_content = '''"""
Step07 Enhanced Matrix Operations - Simplified with Fixed Imports

This is a simplified version that addresses the import issues
identified in the audit while maintaining core functionality.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Use the import fix module
try:
    from src.utils.step07_import_fix import (
        numpy as np, pandas as pd, torch, numba, psutil,
        system_logger, handles_errors, BaseStep, check_dependencies
    )
except ImportError:
    # Fallback imports
    import logging
    system_logger = logging.getLogger('step07_simplified')
    logging.basicConfig(level=logging.INFO)
    
    def handles_errors(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    class BaseStep:
        def __init__(self, config, step_id, step_name):
            self.config = config
            self.step_id = step_id
            self.step_name = step_name

class SimplifiedMatrixOperationsStep(BaseStep):
    """Simplified Step07 with fixed imports and reduced complexity."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, '07', 'simplified_matrix_operations')
        self.logger = system_logger.getChild('SimplifiedMatrixOperations')
        
        # Check dependencies
        if not check_dependencies():
            self.logger.warning("⚠️ Some dependencies missing, using fallback implementations")
        
        # Configuration
        self.matrix_config = config.get('matrix_operations_config', {
            'use_gpu': False,  # Disable GPU by default to avoid torch issues
            'use_numba': False,  # Disable numba by default
            'batch_size': 1000,
            'max_memory_mb': 1024
        })
    
    @handles_errors(exceptions=(Exception,), default_return={'success': False})
    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute simplified matrix operations."""
        self.logger.info("🔢 Starting simplified matrix operations...")
        
        try:
            # Get data
            data_dict = self._get_data_to_process(pipeline_state)
            if not data_dict:
                self.logger.error("❌ No data available for processing")
                return pipeline_state
            
            # Process each split
            matrix_results = {}
            for split_name, data in data_dict.items():
                self.logger.info(f"🧮 Processing {split_name} split...")
                matrices = await self._compute_matrices_simple(data)
                matrix_results[split_name] = matrices
            
            # Update pipeline state
            pipeline_state.update({
                'matrix_results': matrix_results,
                'step07_simplified_completed': True
            })
            
            self.logger.info("✅ Simplified matrix operations completed")
            return pipeline_state
            
        except Exception as e:
            self.logger.error(f"❌ Error in simplified matrix operations: {e}")
            return pipeline_state
    
    def _get_data_to_process(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get data to process with fallback handling."""
        # Try to get engineered data
        if 'engineered_data' in pipeline_state:
            return pipeline_state['engineered_data']
        
        # Try to get advanced features
        if 'advanced_features' in pipeline_state:
            advanced_features = pipeline_state['advanced_features']
            data_dict = {}
            
            for split in ['train', 'val', 'test']:
                if split in advanced_features:
                    path = advanced_features[split]
                    if isinstance(path, str) and Path(path).exists():
                        try:
                            if pd is not None:
                                data_dict[split] = pd.read_parquet(path)
                            else:
                                self.logger.warning(f"⚠️ pandas not available, cannot load {split} data")
                        except Exception as e:
                            self.logger.warning(f"⚠️ Failed to load {split} data: {e}")
            
            if data_dict:
                return data_dict
        
        # Fallback to individual data keys
        data_dict = {}
        for split in ['train', 'val', 'test']:
            if f'{split}_data' in pipeline_state:
                data_dict[split] = pipeline_state[f'{split}_data']
        
        return data_dict
    
    async def _compute_matrices_simple(self, data: Any) -> Dict[str, Any]:
        """Compute matrices with simplified approach."""
        matrices = {}
        
        try:
            if pd is not None and isinstance(data, pd.DataFrame):
                # Get numeric columns
                numeric_cols = data.select_dtypes(include=['number']).columns
                if len(numeric_cols) == 0:
                    self.logger.warning("⚠️ No numeric columns found")
                    return matrices
                
                numeric_data = data[numeric_cols]
                
                # Compute correlation matrix
                try:
                    corr_matrix = numeric_data.corr()
                    matrices['correlation_matrix'] = corr_matrix.values if hasattr(corr_matrix, 'values') else corr_matrix
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to compute correlation matrix: {e}")
                
                # Compute covariance matrix
                try:
                    cov_matrix = numeric_data.cov()
                    matrices['covariance_matrix'] = cov_matrix.values if hasattr(cov_matrix, 'values') else cov_matrix
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to compute covariance matrix: {e}")
                
                # Compute basic statistics
                matrices['feature_stats'] = {
                    'mean': numeric_data.mean().to_dict(),
                    'std': numeric_data.std().to_dict(),
                    'count': numeric_data.count().to_dict()
                }
                
            else:
                self.logger.warning("⚠️ Data is not a pandas DataFrame, skipping matrix computations")
                
        except Exception as e:
            self.logger.error(f"❌ Error in matrix computation: {e}")
        
        return matrices
    
    def validate_inputs(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate inputs with simplified checks."""
        errors = []
        
        # Check for data
        has_data = (
            'engineered_data' in pipeline_state or
            'advanced_features' in pipeline_state or
            any(f'{split}_data' in pipeline_state for split in ['train', 'val', 'test'])
        )
        
        if not has_data:
            errors.append('No data available for processing')
        
        return len(errors) == 0, errors
    
    def get_required_inputs(self) -> List[str]:
        """Get required inputs."""
        return ['engineered_data or split data']
    
    def get_produced_outputs(self) -> List[str]:
        """Get produced outputs."""
        return ['matrix_results']

# Factory function for creating the step
def create_step07_step(config: Dict[str, Any]) -> SimplifiedMatrixOperationsStep:
    """Create a Step07 step instance."""
    return SimplifiedMatrixOperationsStep(config)

if __name__ == "__main__":
    # Test the simplified step
    print("🧪 Testing Simplified Step07")
    print("=" * 40)
    
    config = {
        'matrix_operations_config': {
            'use_gpu': False,
            'use_numba': False,
            'batch_size': 1000
        }
    }
    
    step = create_step07_step(config)
    print(f"✅ Created step: {step.step_name}")
    print(f"📊 Required inputs: {step.get_required_inputs()}")
    print(f"📤 Produced outputs: {step.get_produced_outputs()}")
'''
    
    with open('src/training/steps/model_training/step07_simplified_fixed.py', 'w') as f:
        f.write(simplified_content)
    
    print("✅ Created src/training/steps/model_training/step07_simplified_fixed.py")

def create_installation_script():
    """Create installation script for different environments."""
    install_script_content = '''#!/bin/bash
# Step07 Dependency Installation Script

echo "🚀 Step07 Dependency Installation"
echo "=================================="

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
    PIP_CMD="pip"
elif command -v conda &> /dev/null; then
    echo "✅ Conda detected"
    PIP_CMD="conda install -c conda-forge"
else
    echo "⚠️ No virtual environment detected"
    echo "Creating virtual environment..."
    python3 -m venv step07_env
    source step07_env/bin/activate
    PIP_CMD="pip"
fi

echo "📦 Installing dependencies..."

# Install core dependencies
$PIP_CMD install numpy>=1.21.0
$PIP_CMD install pandas>=1.3.0
$PIP_CMD install scikit-learn>=1.0.0
$PIP_CMD install scipy>=1.7.0
$PIP_CMD install psutil>=5.8.0

# Install optional dependencies
echo "📦 Installing optional dependencies..."
$PIP_CMD install torch>=1.12.0 || echo "⚠️ PyTorch installation failed, continuing..."
$PIP_CMD install numba>=0.56.0 || echo "⚠️ Numba installation failed, continuing..."
$PIP_CMD install lightgbm>=3.3.0 || echo "⚠️ LightGBM installation failed, continuing..."

echo "🧪 Testing installation..."
python3 -c "
import sklearn
import scipy
import psutil
print('✅ Core dependencies working')

try:
    import torch
    print('✅ PyTorch working')
except ImportError:
    print('⚠️ PyTorch not available')

try:
    import numba
    print('✅ Numba working')
except ImportError:
    print('⚠️ Numba not available')
"

echo "✅ Installation complete!"
echo "Run 'python3 step07_import_verification.py' to verify"
'''
    
    with open('install_step07_dependencies.sh', 'w') as f:
        f.write(install_script_content)
    
    # Make it executable
    os.chmod('install_step07_dependencies.sh', 0o755)
    
    print("✅ Created install_step07_dependencies.sh")

def main():
    """Main function to create all dependency and import fix files."""
    print("🔧 Step07 Dependency and Import Fix")
    print("=" * 50)
    
    # Check environment
    check_python_environment()
    print()
    
    # Create dependency files
    print("📦 Creating dependency management files...")
    create_requirements_file()
    create_conda_environment_file()
    create_dockerfile()
    create_installation_script()
    print()
    
    # Create import fix files
    print("🔧 Creating import fix files...")
    create_import_fix_module()
    create_simplified_step07()
    print()
    
    print("✅ All files created successfully!")
    print()
    print("📋 Next Steps:")
    print("1. Run: chmod +x install_step07_dependencies.sh")
    print("2. Run: ./install_step07_dependencies.sh")
    print("3. Or use conda: conda env create -f environment_step07.yml")
    print("4. Or use Docker: docker build -f Dockerfile.step07 -t step07 .")
    print("5. Test: python3 step07_import_verification.py")

if __name__ == "__main__":
    main()