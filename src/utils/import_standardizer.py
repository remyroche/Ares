from src.utils.tprint import tprint

"""
Import standardization utilities for consistent import patterns.
"""

import ast
import os
import re
from typing import List, Dict, Set, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ImportStandardizer:
    """
    Standardizes import patterns across the codebase.
    """
    
    def __init__(self):
        self.import_order = [
            # Standard library imports
            'builtin',
            # Third-party imports
            'third_party',
            # Local imports
            'local'
        ]
        
        self.standard_library_modules = {
            'os', 'sys', 'json', 'datetime', 'time', 'logging', 'pathlib',
            'typing', 'dataclasses', 'enum', 'collections', 'itertools',
            'functools', 'contextlib', 'asyncio', 'threading', 'multiprocessing',
            'subprocess', 'shutil', 'tempfile', 'uuid', 'hashlib', 'base64',
            'urllib', 'http', 'socket', 'ssl', 'email', 'html', 'xml',
            'csv', 'configparser', 'argparse', 'getopt', 'signal', 'warnings',
            'traceback', 'inspect', 'importlib', 'pkgutil', 'abc', 'weakref',
            'copy', 'pickle', 'sqlite3', 're', 'math', 'statistics', 'random',
            'decimal', 'fractions', 'operator', 'functools', 'itertools'
        }
        
        self.third_party_modules = {
            'numpy', 'pandas', 'scipy', 'sklearn', 'tensorflow', 'torch',
            'matplotlib', 'seaborn', 'plotly', 'requests', 'aiohttp',
            'asyncio', 'pydantic', 'fastapi', 'flask', 'django', 'sqlalchemy',
            'alembic', 'redis', 'celery', 'pytest', 'black', 'flake8',
            'mypy', 'isort', 'pre_commit', 'click', 'rich', 'typer',
            'pydantic', 'marshmallow', 'jsonschema', 'pyyaml', 'toml',
            'python_dotenv', 'structlog', 'loguru', 'prometheus_client',
            'mlflow', 'wandb', 'optuna', 'hyperopt', 'xgboost', 'lightgbm',
            'catboost', 'shap', 'lime', 'eli5', 'boruta', 'featuretools',
            'ta', 'pandas_ta', 'yfinance', 'ccxt', 'binance', 'kraken',
            'coinbase', 'alpaca', 'ibapi', 'quantlib', 'zipline', 'backtrader',
            'vectorbt', 'empyrical', 'pyfolio', 'bt', 'ffn', 'ta_lib'
        }
    
    def standardize_file_imports(self, file_path: str) -> bool:
        """
        Standardize imports in a single file.
        
        Args:
            file_path: Path to the file to standardize
            
        Returns:
            True if file was modified, False otherwise
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            original_imports = self._extract_imports(tree)
            
            if not original_imports:
                return False
            
            standardized_imports = self._standardize_imports(original_imports)
            
            if standardized_imports == original_imports:
                return False
            
            # Replace imports in content
            new_content = self._replace_imports_in_content(content, original_imports, standardized_imports)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            logger.info(f"Standardized imports in {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error standardizing imports in {file_path}: {e}")
            return False
    
    def _extract_imports(self, tree: ast.AST) -> List[Dict]:
        """Extract import statements from AST."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        'type': 'import',
                        'module': alias.name,
                        'alias': alias.asname,
                        'line': node.lineno
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append({
                        'type': 'from_import',
                        'module': module,
                        'name': alias.name,
                        'alias': alias.asname,
                        'line': node.lineno
                    })
        
        return imports
    
    def _standardize_imports(self, imports: List[Dict]) -> List[Dict]:
        """Standardize import order and formatting."""
        # Group imports by category
        grouped_imports = {
            'builtin': [],
            'third_party': [],
            'local': []
        }
        
        for imp in imports:
            category = self._categorize_import(imp)
            grouped_imports[category].append(imp)
        
        # Sort each category
        for category in grouped_imports:
            grouped_imports[category].sort(key=lambda x: (x['module'], x.get('name', '')))
        
        # Flatten back to list
        standardized = []
        for category in self.import_order:
            standardized.extend(grouped_imports[category])
        
        return standardized
    
    def _categorize_import(self, imp: Dict) -> str:
        """Categorize an import as builtin, third_party, or local."""
        module = imp['module']
        
        if not module:
            return 'local'
        
        # Check if it's a standard library module
        if module.split('.')[0] in self.standard_library_modules:
            return 'builtin'
        
        # Check if it's a third-party module
        if module.split('.')[0] in self.third_party_modules:
            return 'third_party'
        
        # Check if it's a local import (starts with src or relative)
        if module.startswith('src.') or module.startswith('.'):
            return 'local'
        
        # Default to third_party for unknown modules
        return 'third_party'
    
    def _replace_imports_in_content(self, content: str, original_imports: List[Dict], 
                                  standardized_imports: List[Dict]) -> str:
        """Replace imports in file content."""
        lines = content.split('\n')
        
        # Find import lines
        import_lines = set()
        for imp in original_imports:
            import_lines.add(imp['line'] - 1)  # Convert to 0-based index
        
        # Remove original import lines
        new_lines = []
        for i, line in enumerate(lines):
            if i not in import_lines:
                new_lines.append(line)
        
        # Generate new import statements
        new_imports = self._generate_import_statements(standardized_imports)
        
        # Find insertion point (after docstring and before other code)
        insertion_point = 0
        for i, line in enumerate(new_lines):
            if line.strip() and not line.strip().startswith(('"""', "'''", '#')):
                insertion_point = i
                break
        
        # Insert new imports
        new_lines[insertion_point:insertion_point] = new_imports + ['']
        
        return '\n'.join(new_lines)
    
    def _generate_import_statements(self, imports: List[Dict]) -> List[str]:
        """Generate standardized import statements."""
        statements = []
        current_category = None
        
        for imp in imports:
            category = self._categorize_import(imp)
            
            # Add blank line between categories
            if current_category and current_category != category:
                statements.append('')
            
            current_category = category
            
            # Generate import statement
            if imp['type'] == 'import':
                if imp['alias']:
                    statements.append(f"import {imp['module']} as {imp['alias']}")
                else:
                    statements.append(f"import {imp['module']}")
            else:  # from_import
                if imp['alias']:
                    statements.append(f"from {imp['module']} import {imp['name']} as {imp['alias']}")
                else:
                    statements.append(f"from {imp['module']} import {imp['name']}")
        
        return statements
    
    def standardize_directory_imports(self, directory: str) -> Dict[str, int]:
        """
        Standardize imports in all Python files in a directory.
        
        Args:
            directory: Directory to process
            
        Returns:
            Dictionary with statistics about processed files
        """
        stats = {
            'total_files': 0,
            'modified_files': 0,
            'errors': 0
        }
        
        for root, dirs, files in os.walk(directory):
            # Skip __pycache__ directories
            dirs[:] = [d for d in dirs if d != '__pycache__']
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    stats['total_files'] += 1
                    
                    try:
                        if self.standardize_file_imports(file_path):
                            stats['modified_files'] += 1
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {e}")
                        stats['errors'] += 1
        
        return stats


def standardize_imports_in_file(file_path: str) -> bool:
    """Standardize imports in a single file."""
    standardizer = ImportStandardizer()
    return standardizer.standardize_file_imports(file_path)


def standardize_imports_in_directory(directory: str) -> Dict[str, int]:
    """Standardize imports in all Python files in a directory."""
    standardizer = ImportStandardizer()
    return standardizer.standardize_directory_imports(directory)


# Example usage
if __name__ == "__main__":
    # Standardize imports in src directory
    stats = standardize_imports_in_directory("src")
    tprint(f"Processed {stats['total_files']} files")
    tprint(f"Modified {stats['modified_files']} files")
    tprint(f"Errors: {stats['errors']}")
