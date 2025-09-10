#!/usr/bin/env python3
"""
Manual verification of unused files to identify truly dead code.
"""

import json
import re
import ast
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

def extract_unused_files() -> List[Dict]:
    """Extract unused files from import analysis results."""
    json_file = Path("/Users/remyroche/Documents/Ares/imports_analysis_20250910_193741/import_verifier_analysis.json")

    with open(json_file, 'r') as f:
        data = json.load(f)

    unused_files = []
    for file_path, info in data['import_status'].items():
        if not info.get('is_imported', True):
            unused_files.append({
                'path': file_path,
                'module_name': info.get('module_name', ''),
                'import_count': info.get('import_count', 0)
            })

    return unused_files

def find_all_python_files(repo_root: Path) -> List[Path]:
    """Find all Python files in the repository."""
    python_files = []
    for py_file in repo_root.rglob("*.py"):
        # Skip problematic directories
        if any(excluded in py_file.parts for excluded in ["venv", "__pycache__", ".git", "node_modules", ".pytest_cache"]):
            continue
        python_files.append(py_file)
    return python_files

def search_dynamic_imports(content: str, module_name: str) -> List[Dict]:
    """Search for dynamic imports of the module."""
    findings = []

    # Check for __import__ calls
    import_pattern = r'__import__\s*\(\s*[\'"]([^\'"]*' + re.escape(module_name) + r'[^\'"]*)[\'"]'
    for match in re.finditer(import_pattern, content):
        findings.append({
            'type': 'dynamic_import',
            'pattern': '__import__',
            'match': match.group(0),
            'line': content[:match.start()].count('\n') + 1
        })

    # Check for importlib calls
    importlib_pattern = r'importlib\.import_module\s*\(\s*[\'"]([^\'"]*' + re.escape(module_name) + r'[^\'"]*)[\'"]'
    for match in re.finditer(importlib_pattern, content):
        findings.append({
            'type': 'dynamic_import',
            'pattern': 'importlib.import_module',
            'match': match.group(0),
            'line': content[:match.start()].count('\n') + 1
        })

    return findings

def search_string_references(content: str, module_name: str, file_name: str) -> List[Dict]:
    """Search for string references to the module name."""
    findings = []

    # Search for module name as string
    module_pattern = r'[\'"]' + re.escape(module_name) + r'[\'"]'
    for match in re.finditer(module_pattern, content):
        findings.append({
            'type': 'string_reference',
            'pattern': 'module_name_string',
            'match': match.group(0),
            'line': content[:match.start()].count('\n') + 1
        })

    # Search for file path references
    file_pattern = r'[\'"]' + re.escape(file_name) + r'[\'"]'
    for match in re.finditer(file_pattern, content):
        findings.append({
            'type': 'file_reference',
            'pattern': 'file_path_string',
            'match': match.group(0),
            'line': content[:match.start()].count('\n') + 1
        })

    return findings

def search_class_references(content: str, file_path: Path) -> List[Dict]:
    """Search for class name references that might indicate usage."""
    findings = []

    try:
        tree = ast.parse(content)
        classes = []
        functions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.FunctionDef):
                functions.append(node.name)

        # Search for references to these classes/functions
        all_names = classes + functions
        for name in all_names:
            name_pattern = r'\b' + re.escape(name) + r'\b'
            for match in re.finditer(name_pattern, content):
                # Get the line content around the match
                lines = content.split('\n')
                line_num = content[:match.start()].count('\n')
                if line_num < len(lines):
                    line_content = lines[line_num].strip()
                    # Skip the definition itself
                    if f'class {name}' not in line_content and f'def {name}' not in line_content:
                        findings.append({
                            'type': 'name_reference',
                            'name': name,
                            'match': match.group(0),
                            'line': line_num + 1
                        })

    except SyntaxError:
        pass

    return findings

def verify_file_usage(unused_file: Dict, all_python_files: List[Path]) -> Dict:
    """Verify if an unused file is truly dead by checking all possible references."""
    file_path = Path(unused_file['path'])
    module_name = unused_file['module_name']
    file_name = file_path.name

    print(f"\n🔍 Verifying: {file_path}")
    print(f"   Module: {module_name}")
    print(f"   File: {file_name}")

    # Read the content of the unused file to understand what it exports
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            unused_content = f.read()
    except Exception as e:
        return {
            'file': str(file_path),
            'status': 'error',
            'reason': f'Cannot read file: {e}',
            'is_dead': False
        }

    total_findings = []
    files_with_references = []

    # Check each Python file in the repository
    for other_file in all_python_files:
        if other_file == file_path:
            continue

        try:
            with open(other_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            continue

        findings = []

        # Search for direct imports (this was already done by the analyzer)
        # But let's double-check for any we might have missed

        # Search for dynamic imports
        dynamic_findings = search_dynamic_imports(content, module_name)
        findings.extend(dynamic_findings)

        # Search for string references
        string_findings = search_string_references(content, module_name, file_name)
        findings.extend(string_findings)

        # Search for class/function name references
        name_findings = search_class_references(content, file_path)
        findings.extend(name_findings)

        if findings:
            total_findings.extend(findings)
            files_with_references.append({
                'file': str(other_file),
                'findings': findings
            })

    # Analyze the findings
    is_dead = len(total_findings) == 0

    result = {
        'file': str(file_path),
        'module_name': module_name,
        'is_dead': is_dead,
        'total_references': len(total_findings),
        'files_with_references': len(files_with_references),
        'references': total_findings[:10],  # Limit for readability
        'referencing_files': [f['file'] for f in files_with_references[:5]]  # Limit for readability
    }

    if is_dead:
        print("   ✅ VERDICT: TRULY DEAD - No references found")
    else:
        print(f"   ⚠️  VERDICT: NOT DEAD - Found {len(total_findings)} references in {len(files_with_references)} files")
        if files_with_references:
            print(f"   📁 Top referencing files: {files_with_references[0]['file']}")

    return result

def main():
    """Main verification process."""
    print("🔬 MANUAL VERIFICATION OF UNUSED FILES")
    print("=" * 80)

    # Get list of unused files
    unused_files = extract_unused_files()
    print(f"Found {len(unused_files)} unused files to verify")

    # Get all Python files for comprehensive search
    repo_root = Path("/Users/remyroche/Documents/Ares")
    all_python_files = find_all_python_files(repo_root)
    print(f"Repository contains {len(all_python_files)} Python files")

    # Verify each unused file
    results = []
    dead_files = []
    not_dead_files = []

    for i, unused_file in enumerate(unused_files, 1):
        print(f"\n[{i:3d}/{len(unused_files)}] ", end="")
        result = verify_file_usage(unused_file, all_python_files)
        results.append(result)

        if result['is_dead']:
            dead_files.append(result)
        else:
            not_dead_files.append(result)

    # Generate summary
    print("\n📊 VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"Total files verified: {len(results)}")
    print(f"Truly dead files: {len(dead_files)}")
    print(f"Files with references: {len(not_dead_files)}")

    # Save detailed results
    output_file = repo_root / "manual_verification_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': str(Path(__file__).stat().st_mtime),
            'summary': {
                'total_verified': len(results),
                'dead_files': len(dead_files),
                'not_dead_files': len(not_dead_files)
            },
            'dead_files': dead_files,
            'not_dead_files': not_dead_files,
            'detailed_results': results
        }, f, indent=2)

    print(f"\n📁 Detailed results saved to: {output_file}")

    # Print dead files
    if dead_files:
        print("\n✅ TRULY DEAD FILES:")
        for file in dead_files[:20]:  # Show first 20
            print(f"  • {file['file']}")

        if len(dead_files) > 20:
            print(f"  ... and {len(dead_files) - 20} more")

    # Print files with references
    if not_dead_files:
        print("\n⚠️  FILES WITH REFERENCES:")
        for file in not_dead_files[:10]:  # Show first 10
            print(f"  • {file['file']} ({file['total_references']} references)")

        if len(not_dead_files) > 10:
            print(f"  ... and {len(not_dead_files) - 10} more")

if __name__ == "__main__":
    main()
