#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Auto Dependency Installer

Automatically detects and installs missing Python dependencies based on import errors.
"""

import ast
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
import json
from datetime import datetime

class AutoDependencyInstaller:
    """Automatically install missing Python dependencies."""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.missing_dependencies = set()
        self.installed_dependencies = set()
        self.failed_installations = set()
        
        # Common package mappings
        self.package_mappings = {
            'numpy': 'numpy',
            'pandas': 'pandas', 
            'sklearn': 'scikit-learn',
            'torch': 'torch',
            'tensorflow': 'tensorflow',
            'matplotlib': 'matplotlib',
            'seaborn': 'seaborn',
            'plotly': 'plotly',
            'requests': 'requests',
            'celery': 'celery',
            'redis': 'redis',
            'psycopg2': 'psycopg2-binary',
            'pymongo': 'pymongo',
            'sqlalchemy': 'sqlalchemy',
            'flask': 'flask',
            'django': 'django',
            'fastapi': 'fastapi',
            'uvicorn': 'uvicorn',
            'pytest': 'pytest',
            'black': 'black',
            'isort': 'isort',
            'flake8': 'flake8',
            'mypy': 'mypy',
            'optuna': 'optuna',
            'lightgbm': 'lightgbm',
            'xgboost': 'xgboost',
            'shap': 'shap',
            'joblib': 'joblib',
            'psutil': 'psutil',
            'networkx': 'networkx',
            'astroid': 'astroid',
            'pylint': 'pylint'
        }
    
    def extract_imports_from_file(self, file_path: str) -> Set[str]:
        """Extract all import statements from a Python file."""
        imports = set()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])
                        
        except Exception as e:
            tprint(f"Warning: Could not parse {file_path}: {e}")
            
        return imports
    
    def scan_project_imports(self, pattern: str = "**/*.py") -> Set[str]:
        """Scan all Python files in project for imports."""
        all_imports = set()
        python_files = list(self.project_root.glob(pattern))
        
        tprint(f"🔍 Scanning {len(python_files)} Python files for imports...")
        
        for file_path in python_files:
            imports = self.extract_imports_from_file(str(file_path))
            all_imports.update(imports)
        
        return all_imports
    
    def check_installed_packages(self) -> Set[str]:
        """Check which packages are currently installed."""
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                                  capture_output=True, text=True)
            installed = set()
            for line in result.stdout.split('\n')[2:]:  # Skip header
                if line.strip():
                    package = line.split()[0].lower()
                    installed.add(package)
            return installed
        except Exception as e:
            tprint(f"Warning: Could not check installed packages: {e}")
            return set()
    
    def identify_missing_dependencies(self) -> Set[str]:
        """Identify missing dependencies by comparing imports with installed packages."""
        all_imports = self.scan_project_imports()
        installed_packages = self.check_installed_packages()
        
        missing = set()
        for import_name in all_imports:
            # Skip standard library modules
            if import_name in ['os', 'sys', 'json', 'datetime', 'pathlib', 'typing', 
                             'collections', 'itertools', 'functools', 'operator',
                             'math', 'random', 'string', 're', 'urllib', 'http',
                             'email', 'html', 'xml', 'csv', 'sqlite3', 'threading',
                             'multiprocessing', 'asyncio', 'logging', 'warnings',
                             'unittest', 'doctest', 'pickle', 'copy', 'shutil',
                             'tempfile', 'glob', 'fnmatch', 'linecache', 'traceback',
                             'inspect', 'gc', 'weakref', 'types', 'abc', 'enum',
                             'dataclasses', 'contextlib', 'functools', 'itertools']:
                continue
                
            # Check if package is installed
            package_name = self.package_mappings.get(import_name, import_name)
            if package_name not in installed_packages:
                missing.add(package_name)
        
        return missing
    
    def install_dependency(self, package: str) -> bool:
        """Install a single dependency."""
        try:
            tprint(f"📦 Installing {package}...")
            result = subprocess.run([sys.executable, '-m', 'pip', 'install', 
                                   '--break-system-packages', package], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                tprint(f"✅ Successfully installed {package}")
                self.installed_dependencies.add(package)
                return True
            else:
                tprint(f"❌ Failed to install {package}: {result.stderr}")
                self.failed_installations.add(package)
                return False
                
        except Exception as e:
            tprint(f"❌ Error installing {package}: {e}")
            self.failed_installations.add(package)
            return False
    
    def install_all_missing_dependencies(self, dry_run: bool = False) -> Dict[str, any]:
        """Install all missing dependencies."""
        missing = self.identify_missing_dependencies()
        
        if not missing:
            tprint("✅ No missing dependencies found!")
            return {"status": "success", "installed": [], "failed": []}
        
        tprint(f"🔍 Found {len(missing)} missing dependencies:")
        for dep in sorted(missing):
            tprint(f"  - {dep}")
        
        if dry_run:
            tprint("\n🔍 DRY RUN - Would install:")
            for dep in sorted(missing):
                tprint(f"  pip install {dep}")
            return {"status": "dry_run", "would_install": list(missing)}
        
        tprint(f"\n📦 Installing {len(missing)} dependencies...")
        
        for package in sorted(missing):
            self.install_dependency(package)
        
        return {
            "status": "completed",
            "installed": list(self.installed_dependencies),
            "failed": list(self.failed_installations),
            "total_missing": len(missing)
        }
    
    def generate_requirements_txt(self, output_file: str = None) -> str:
        """Generate a requirements.txt file with all dependencies."""
        if not output_file:
            output_file = str(self.project_root / "requirements_auto_generated.txt")
        
        all_imports = self.scan_project_imports()
        requirements = set()
        
        for import_name in all_imports:
            package_name = self.package_mappings.get(import_name, import_name)
            requirements.add(package_name)
        
        with open(output_file, 'w') as f:
            for req in sorted(requirements):
                f.write(f"{req}\n")
        
        tprint(f"📄 Generated requirements file: {output_file}")
        return output_file

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto Dependency Installer")
    parser.add_argument("--project-root", "-p", help="Project root directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be installed")
    parser.add_argument("--generate-requirements", action="store_true", help="Generate requirements.txt")
    parser.add_argument("--output", "-o", help="Output file for requirements.txt")
    
    args = parser.parse_args()
    
    installer = AutoDependencyInstaller(args.project_root)
    
    if args.generate_requirements:
        installer.generate_requirements_txt(args.output)
    
    result = installer.install_all_missing_dependencies(args.dry_run)
    
    tprint(f"\n📊 SUMMARY:")
    tprint(f"  Status: {result['status']}")
    if 'installed' in result:
        tprint(f"  Installed: {len(result['installed'])} packages")
    if 'failed' in result:
        tprint(f"  Failed: {len(result['failed'])} packages")
    if 'total_missing' in result:
        tprint(f"  Total missing: {result['total_missing']} packages")

if __name__ == "__main__":
    main()