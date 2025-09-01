#!/usr/bin/env python3
"""
Batch processor to find and remove unused imports across many files
"""

import ast
import sys


def is_import_used(import_name, content, ast_tree):
    """Check if an import is actually used in the code."""
    # Check if used as a name
    for node in ast.walk(ast_tree):
        if isinstance(node, ast.Name) and node.id == import_name:
            return True
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == import_name:
            return True
    
    # Check if used in strings (like type annotations, etc.)
    if f"'{import_name}'" in content or f'"{import_name}"' in content:
        return True
    
    # Check for indirect usage patterns
    if f"{import_name}." in content:
        return True
    
    return False


def find_and_remove_unused_imports(filepath, dry_run=True):
    """Find and remove unused imports from a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        lines = content.split('\n')
        imports_to_remove = []
        
        # Find all import statements
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_name = alias.asname or alias.name.split('.')[0]
                    if not is_import_used(import_name, content, tree):
                        imports_to_remove.append(node.lineno - 1)  # 0-based index
            elif isinstance(node, ast.ImportFrom):
                # For from imports, check if any of the imported names are used
                unused_names = []
                for alias in node.names:
                    import_name = alias.asname or alias.name
                    if import_name != '*' and not is_import_used(import_name, content, tree):
                        unused_names.append(alias.name)
                
                # If all names in the from import are unused, mark the whole line
                if len(unused_names) == len(node.names) and node.names[0].name != '*':
                    imports_to_remove.append(node.lineno - 1)
        
        if not imports_to_remove:
            return False
        
        if dry_run:
            print(f"\n{filepath}:")
            for line_idx in sorted(set(imports_to_remove)):
                if line_idx < len(lines):
                    print(f"  Would remove line {line_idx + 1}: {lines[line_idx].strip()}")
        else:
            # Remove imports in reverse order to maintain line numbers
            for line_idx in sorted(set(imports_to_remove), reverse=True):
                if line_idx < len(lines):
                    print(f"Removing line {line_idx + 1}: {lines[line_idx].strip()}")
                    lines.pop(line_idx)
            
            # Write back the file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
        
        return True
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False


def process_files(file_pattern, dry_run=True):
    """Process multiple files matching a pattern."""
    from glob import glob
    
    files = glob(file_pattern)
    total_files = len(files)
    processed = 0
    
    print(f"Processing {total_files} files matching '{file_pattern}'...")
    
    for filepath in files:
        # Skip files that are likely to have syntax errors
        if any(skip in filepath for skip in ['__pycache__', '.git', 'test_results', 'log/']):
            continue
            
        try:
            # Quick syntax check
            with open(filepath, 'r', encoding='utf-8') as f:
                ast.parse(f.read())
            
            if find_and_remove_unused_imports(filepath, dry_run):
                processed += 1
                
        except SyntaxError:
            print(f"Skipping {filepath} (syntax error)")
        except Exception as e:
            print(f"Skipping {filepath} ({e})")
    
    print(f"\nProcessed {processed} files with unused imports.")


if __name__ == '__main__':
    dry_run = '--dry-run' in sys.argv or len(sys.argv) < 2
    
    if len(sys.argv) < 2:
        # Default to processing some common patterns
        patterns = [
            "*.py",
            "src/**/*.py", 
            "scripts/*.py"
        ]
    else:
        patterns = sys.argv[1:]
        patterns = [p for p in patterns if p != '--dry-run']
    
    print(f"{'DRY RUN: ' if dry_run else ''}Cleaning unused imports...")
    
    for pattern in patterns:
        process_files(pattern, dry_run)