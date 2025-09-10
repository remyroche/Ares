#!/usr/bin/env python3
"""
Debug script to check what imports are being detected.
"""

import ast
from pathlib import Path

def debug_imports(file_path):
    """Debug imports in a specific file."""
    print(f"Debugging imports in: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
    
    try:
        tree = ast.parse(content)
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return
    
    imports = set()
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
                print(f"Import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
                print(f"ImportFrom module: {node.module}")
                
                # Add specific imports from the module
                for alias in node.names:
                    if alias.name != '*':
                        full_import = f"{node.module}.{alias.name}"
                        imports.add(full_import)
                        print(f"ImportFrom specific: {full_import}")
    
    print(f"\nTotal imports found: {len(imports)}")
    print("All imports:")
    for imp in sorted(imports):
        print(f"  {imp}")
    
    # Check for sr_clustering imports specifically
    sr_clustering_imports = [imp for imp in imports if 'sr_clustering' in imp]
    print(f"\nSR clustering related imports: {sr_clustering_imports}")

if __name__ == "__main__":
    # Debug the step02_5 file
    step02_5_file = "/workspace/src/training/steps/data_collection/data_preparation/step02_5_sr_optimization.py"
    debug_imports(step02_5_file)