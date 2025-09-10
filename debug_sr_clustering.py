#!/usr/bin/env python3
"""
Debug script to specifically check sr_clustering imports.
"""

import ast
from pathlib import Path

def debug_sr_clustering_imports():
    """Debug sr_clustering imports specifically."""
    project_root = Path("/workspace")
    
    # Get the step02_5 file
    step02_5_file = project_root / "src/training/steps/data_collection/data_preparation/step02_5_sr_optimization.py"
    
    print(f"Debugging imports in: {step02_5_file}")
    
    try:
        with open(step02_5_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {step02_5_file}: {e}")
        return
    
    try:
        tree = ast.parse(content)
    except Exception as e:
        print(f"Error parsing {step02_5_file}: {e}")
        return
    
    imports = set()
    
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
    
    print(f"\nAll imports found: {len(imports)}")
    
    # Check for sr_clustering specifically
    sr_clustering_imports = [imp for imp in imports if 'sr_clustering' in imp]
    print(f"\nSR clustering related imports: {len(sr_clustering_imports)}")
    for imp in sr_clustering_imports:
        print(f"  {imp}")
    
    # Now check if the module name matches
    module_name = "src.utils.sr_clustering"
    print(f"\nChecking if module '{module_name}' is imported:")
    print(f"  Direct match: {module_name in imports}")
    
    # Check for prefix matches
    prefix_matches = [imp for imp in imports if imp.startswith(module_name + '.')]
    print(f"  Prefix matches: {len(prefix_matches)}")
    for match in prefix_matches:
        print(f"    {match}")
    
    # Test the corrected logic
    is_imported = module_name in imports or any(imp.startswith(module_name + '.') for imp in imports)
    print(f"  Corrected logic result: {is_imported}")

if __name__ == "__main__":
    debug_sr_clustering_imports()