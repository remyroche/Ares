#!/usr/bin/env python3
"""
Analyze unused scripts in the codebase.
This script identifies Python files that are never imported or referenced.
"""

import ast
import os
import sys
from pathlib import Path
from collections import defaultdict
import re

def get_all_python_files(root_dir):
    """Get all Python files in the directory tree."""
    python_files = []
    for root, dirs, files in os.walk(root_dir):
        # Skip common directories that shouldn't be analyzed
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', 'venv', 'env']]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def extract_imports_from_file(file_path):
    """Extract all imports from a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        imports = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
                    # Also add specific imports
                    for alias in node.names:
                        if alias.name != '*':
                            imports.add(f"{node.module}.{alias.name}")
        
        return imports
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return set()

def get_module_name_from_path(file_path, root_dir):
    """Convert file path to module name."""
    rel_path = os.path.relpath(file_path, root_dir)
    module_path = rel_path.replace(os.sep, '.')
    
    # Remove .py extension
    if module_path.endswith('.py'):
        module_path = module_path[:-3]
    
    # Handle __init__.py files
    if module_path.endswith('.__init__'):
        module_path = module_path[:-9]
    
    return module_path

def is_script_file(file_path):
    """Check if a file looks like a standalone script (has if __name__ == '__main__')."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return 'if __name__' in content and '__main__' in content
    except:
        return False

def is_test_file(file_path):
    """Check if a file is a test file."""
    filename = os.path.basename(file_path)
    return (filename.startswith('test_') or 
            filename.endswith('_test.py') or 
            'test' in filename.lower())

def is_example_file(file_path):
    """Check if a file is an example or demo file."""
    filename = os.path.basename(file_path)
    return (filename.startswith('example_') or 
            filename.startswith('demo_') or
            'example' in filename.lower() or
            'demo' in filename.lower())

def analyze_unused_scripts(root_dir):
    """Analyze which scripts are unused."""
    print(f"🔍 Analyzing unused scripts in {root_dir}")
    
    # Get all Python files
    python_files = get_all_python_files(root_dir)
    print(f"📁 Found {len(python_files)} Python files")
    
    # Extract all imports from all files
    all_imports = set()
    file_imports = {}
    module_names = {}
    
    print("📥 Extracting imports from all files...")
    for file_path in python_files:
        imports = extract_imports_from_file(file_path)
        file_imports[file_path] = imports
        all_imports.update(imports)
        
        # Get module name for this file
        module_name = get_module_name_from_path(file_path, root_dir)
        module_names[file_path] = module_name
    
    print(f"📊 Found {len(all_imports)} unique imports")
    
    # Analyze each file
    unused_files = []
    script_files = []
    test_files = []
    example_files = []
    library_files = []
    
    print("🔍 Analyzing file usage...")
    for file_path in python_files:
        module_name = module_names[file_path]
        filename = os.path.basename(file_path)
        
        # Categorize files
        if is_test_file(file_path):
            test_files.append(file_path)
        elif is_example_file(file_path):
            example_files.append(file_path)
        elif is_script_file(file_path):
            script_files.append(file_path)
        else:
            library_files.append(file_path)
        
        # Check if this module is imported anywhere
        is_imported = False
        for other_file, imports in file_imports.items():
            if other_file != file_path:
                # Check if this module is imported
                if (module_name in imports or 
                    any(imp.startswith(module_name + '.') for imp in imports) or
                    any(imp.endswith('.' + module_name.split('.')[-1]) for imp in imports)):
                    is_imported = True
                    break
        
        # Also check for direct file references (like in subprocess calls)
        is_referenced = False
        for other_file in python_files:
            if other_file != file_path:
                try:
                    with open(other_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    if filename in content or file_path in content:
                        is_referenced = True
                        break
                except:
                    pass
        
        if not is_imported and not is_referenced:
            unused_files.append(file_path)
    
    return {
        'unused_files': unused_files,
        'script_files': script_files,
        'test_files': test_files,
        'example_files': example_files,
        'library_files': library_files,
        'total_files': len(python_files)
    }

def main():
    root_dir = "/workspace"
    
    results = analyze_unused_scripts(root_dir)
    
    print(f"\n{'='*80}")
    print("📊 UNUSED SCRIPTS ANALYSIS RESULTS")
    print(f"{'='*80}")
    
    print(f"\n📁 Total Python files: {results['total_files']}")
    print(f"🔧 Script files: {len(results['script_files'])}")
    print(f"🧪 Test files: {len(results['test_files'])}")
    print(f"📚 Example files: {len(results['example_files'])}")
    print(f"📦 Library files: {len(results['library_files'])}")
    print(f"❌ Potentially unused files: {len(results['unused_files'])}")
    
    if results['unused_files']:
        print(f"\n{'='*80}")
        print("❌ POTENTIALLY UNUSED FILES")
        print(f"{'='*80}")
        
        for i, file_path in enumerate(results['unused_files'][:50], 1):  # Show first 50
            rel_path = os.path.relpath(file_path, root_dir)
            print(f"{i:3d}. {rel_path}")
        
        if len(results['unused_files']) > 50:
            print(f"... and {len(results['unused_files']) - 50} more files")
    
    # Show some examples of each category
    print(f"\n{'='*80}")
    print("📋 FILE CATEGORIES")
    print(f"{'='*80}")
    
    print(f"\n🔧 SCRIPT FILES (first 10):")
    for i, file_path in enumerate(results['script_files'][:10], 1):
        rel_path = os.path.relpath(file_path, root_dir)
        print(f"{i:2d}. {rel_path}")
    
    print(f"\n🧪 TEST FILES (first 10):")
    for i, file_path in enumerate(results['test_files'][:10], 1):
        rel_path = os.path.relpath(file_path, root_dir)
        print(f"{i:2d}. {rel_path}")
    
    print(f"\n📚 EXAMPLE FILES (first 10):")
    for i, file_path in enumerate(results['example_files'][:10], 1):
        rel_path = os.path.relpath(file_path, root_dir)
        print(f"{i:2d}. {rel_path}")
    
    # Save results to file
    with open('/workspace/unused_scripts_analysis.json', 'w') as f:
        import json
        json.dump({
            'unused_files': [os.path.relpath(f, root_dir) for f in results['unused_files']],
            'script_files': [os.path.relpath(f, root_dir) for f in results['script_files']],
            'test_files': [os.path.relpath(f, root_dir) for f in results['test_files']],
            'example_files': [os.path.relpath(f, root_dir) for f in results['example_files']],
            'library_files': [os.path.relpath(f, root_dir) for f in results['library_files']],
            'total_files': results['total_files']
        }, f, indent=2)
    
    print(f"\n💾 Results saved to /workspace/unused_scripts_analysis.json")

if __name__ == "__main__":
    main()