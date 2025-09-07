#!/usr/bin/env python3
"""
Detect files that aren't called/imported anywhere in the codebase
"""

import ast
import os
from pathlib import Path
from collections import defaultdict
import re

def extract_imports_from_file(file_path):
    """Extract all imports from a Python file."""
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
        print(f"Error parsing {file_path}: {e}")
    
    return imports

def find_python_files(root_dir):
    """Find all Python files in the directory tree."""
    python_files = []
    for root, dirs, files in os.walk(root_dir):
        # Skip certain directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', 'venv', 'env']]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    return python_files

def get_module_name(file_path, root_dir):
    """Convert file path to module name."""
    rel_path = os.path.relpath(file_path, root_dir)
    module_name = rel_path.replace(os.sep, '.').replace('.py', '')
    return module_name

def detect_unused_files():
    """Detect files that aren't imported anywhere."""
    root_dir = "/Users/remyroche/Documents/Ares"
    
    print("🔍 Detecting unused files...")
    print("=" * 50)
    
    # Find all Python files
    python_files = find_python_files(root_dir)
    print(f"📊 Found {len(python_files)} Python files")
    
    # Extract imports from all files
    all_imports = set()
    file_imports = {}
    
    for file_path in python_files:
        imports = extract_imports_from_file(file_path)
        file_imports[file_path] = imports
        all_imports.update(imports)
    
    print(f"📊 Found {len(all_imports)} unique imported modules")
    
    # Check which files are never imported
    unused_files = []
    potentially_unused = []
    
    for file_path in python_files:
        module_name = get_module_name(file_path, root_dir)
        
        # Skip certain files that are expected to be unused
        filename = os.path.basename(file_path)
        if filename in ['__init__.py', 'setup.py', 'conftest.py']:
            continue
            
        # Check if this module is imported anywhere
        is_imported = False
        for other_file, imports in file_imports.items():
            if other_file != file_path:
                # Check if any part of the module name is imported
                for imp in imports:
                    if module_name.startswith(imp) or imp in module_name:
                        is_imported = True
                        break
                if is_imported:
                    break
        
        if not is_imported:
            # Check if it's a main script (has if __name__ == "__main__")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if '__name__' in content and '__main__' in content:
                    potentially_unused.append((file_path, "Main script (may be intentionally unused)"))
                else:
                    unused_files.append((file_path, "No imports found"))
            except:
                unused_files.append((file_path, "Error reading file"))
    
    # Report results
    print(f"\n📋 UNUSED FILES ANALYSIS:")
    print(f"   • Total Python files: {len(python_files)}")
    print(f"   • Definitely unused: {len(unused_files)}")
    print(f"   • Potentially unused (main scripts): {len(potentially_unused)}")
    
    if unused_files:
        print(f"\n🚨 DEFINITELY UNUSED FILES ({len(unused_files)}):")
        for file_path, reason in sorted(unused_files):
            rel_path = os.path.relpath(file_path, root_dir)
            print(f"   • {rel_path} - {reason}")
    
    if potentially_unused:
        print(f"\n⚠️  POTENTIALLY UNUSED FILES ({len(potentially_unused)}):")
        for file_path, reason in sorted(potentially_unused):
            rel_path = os.path.relpath(file_path, root_dir)
            print(f"   • {rel_path} - {reason}")
    
    # Check for high-level analysis
    print(f"\n📊 ANALYSIS SUMMARY:")
    print(f"   • Unused file rate: {len(unused_files)/len(python_files)*100:.1f}%")
    print(f"   • Main script rate: {len(potentially_unused)/len(python_files)*100:.1f}%")
    
    if len(unused_files) > len(python_files) * 0.3:
        print("⚠️  High unused file rate detected - consider cleanup")
    elif len(unused_files) < len(python_files) * 0.1:
        print("✅ Low unused file rate - good code organization")
    
    return {
        "total_files": len(python_files),
        "unused_files": unused_files,
        "potentially_unused": potentially_unused,
        "unused_rate": len(unused_files)/len(python_files)*100
    }

if __name__ == "__main__":
    detect_unused_files()
