#!/usr/bin/env python3
"""
Debug the analyzer to see why it's not detecting sr_clustering imports.
"""

import ast
from pathlib import Path
from typing import Set

def debug_analyzer():
    """Debug the analyzer logic."""
    project_root = Path("/workspace")
    
    # Get the step02_5 file
    step02_5_file = project_root / "src/training/steps/data_collection/data_preparation/step02_5_sr_optimization.py"
    
    # Get a sr_clustering file
    sr_clustering_file = project_root / "src/utils/sr_clustering/backtesting_enhanced_clustering.py"
    
    print(f"Step02_5 file: {step02_5_file}")
    print(f"SR clustering file: {sr_clustering_file}")
    
    # Extract imports from step02_5
    step02_5_imports = extract_imports(step02_5_file)
    print(f"\nStep02_5 imports: {len(step02_5_imports)}")
    
    # Get module name for sr_clustering file
    sr_clustering_module = get_module_name(sr_clustering_file, project_root)
    print(f"\nSR clustering module name: {sr_clustering_module}")
    
    # Check if the module is imported
    is_imported = is_module_imported(sr_clustering_module, step02_5_imports)
    print(f"Is sr_clustering imported by step02_5: {is_imported}")
    
    # Show relevant imports
    relevant_imports = [imp for imp in step02_5_imports if 'sr_clustering' in imp]
    print(f"\nRelevant imports in step02_5:")
    for imp in relevant_imports:
        print(f"  {imp}")

def extract_imports(file_path: Path) -> Set[str]:
    """Extract all imports from a Python file."""
    imports = set()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return imports
    
    try:
        tree = ast.parse(content)
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return imports
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
                for alias in node.names:
                    if alias.name != '*':
                        full_import = f"{node.module}.{alias.name}"
                        imports.add(full_import)
    
    return imports

def get_module_name(file_path: Path, project_root: Path) -> str:
    """Convert file path to module name."""
    try:
        relative_path = file_path.relative_to(project_root)
        module_parts = list(relative_path.parts)
        if module_parts[-1].endswith('.py'):
            module_parts[-1] = module_parts[-1][:-3]
        return '.'.join(module_parts)
    except Exception:
        return str(file_path.stem)

def is_module_imported(module_name: str, imports: Set[str]) -> bool:
    """Check if a module is imported."""
    # Direct exact match
    if module_name in imports:
        return True
    
    # Check for specific imports from this module
    for imp in imports:
        if imp.startswith(module_name + '.'):
            return True
    
    return False

if __name__ == "__main__":
    debug_analyzer()