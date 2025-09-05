#!/usr/bin/env python3
"""
Script to analyze and fix missing imports for common operations.
"""

import ast
import json
from collections import defaultdict
from pathlib import Path

# Enhanced function to module mappings with comprehensive coverage
COMMON_IMPORTS = {
    # DateTime operations
    "now": ("datetime", "datetime"),
    "today": ("datetime", "date"),
    "timedelta": ("datetime", "timedelta"),
    "datetime": ("datetime", "datetime"),
    "date": ("datetime", "date"),
    "time": ("datetime", "time"),
    "isoformat": None,  # This is a method, not an import
    "strftime": None,   # This is a method, not an import
    "total_seconds": None,  # This is a method, not an import

    # Pandas operations - Enhanced coverage
    "DataFrame": ("pandas", "pd"),
    "Series": ("pandas", "pd"),
    "read_csv": ("pandas", "pd"),
    "read_parquet": ("pandas", "pd"),
    "read_excel": ("pandas", "pd"),
    "read_json": ("pandas", "pd"),
    "concat": ("pandas", "pd"),
    "merge": ("pandas", "pd"),
    "to_datetime": ("pandas", "pd"),
    "fillna": ("pandas", "pd"),  # Common missing import
    "dropna": ("pandas", "pd"),  # Common missing import
    "groupby": ("pandas", "pd"),  # Common missing import
    "rolling": ("pandas", "pd"),
    "shift": ("pandas", "pd"),
    "diff": ("pandas", "pd"),
    "cumsum": ("pandas", "pd"),
    "cumprod": ("pandas", "pd"),
    "pivot_table": ("pandas", "pd"),
    "melt": ("pandas", "pd"),
    "get_dummies": ("pandas", "pd"),
    "cut": ("pandas", "pd"),
    "qcut": ("pandas", "pd"),
    "crosstab": ("pandas", "pd"),
    "value_counts": ("pandas", "pd"),

    # NumPy operations - Enhanced coverage
    "array": ("numpy", "np"),
    "zeros": ("numpy", "np"),
    "ones": ("numpy", "np"),
    "empty": ("numpy", "np"),
    "full": ("numpy", "np"),
    "arange": ("numpy", "np"),
    "linspace": ("numpy", "np"),
    "logspace": ("numpy", "np"),
    "mean": ("numpy", "np"),
    "std": ("numpy", "np"),
    "var": ("numpy", "np"),
    "sum": ("numpy", "np"),  # Common missing import
    "min": ("numpy", "np"),
    "max": ("numpy", "np"),
    "argmin": ("numpy", "np"),
    "argmax": ("numpy", "np"),
    "nan": ("numpy", "np"),
    "inf": ("numpy", "np"),
    "isnan": ("numpy", "np"),
    "isinf": ("numpy", "np"),
    "isfinite": ("numpy", "np"),
    "abs": ("numpy", "np"),
    "sqrt": ("numpy", "np"),
    "exp": ("numpy", "np"),
    "log": ("numpy", "np"),
    "sin": ("numpy", "np"),
    "cos": ("numpy", "np"),
    "tan": ("numpy", "np"),
    "dot": ("numpy", "np"),
    "transpose": ("numpy", "np"),
    "reshape": ("numpy", "np"),
    "flatten": ("numpy", "np"),
    "concatenate": ("numpy", "np"),
    "stack": ("numpy", "np"),
    "hstack": ("numpy", "np"),
    "vstack": ("numpy", "np"),
    "where": ("numpy", "np"),
    "clip": ("numpy", "np"),
    "round": ("numpy", "np"),

    # Path operations
    "Path": ("pathlib", "Path"),
    "exists": None,  # This is usually a method on Path
    "mkdir": None,   # This is usually a method on Path
    "join": ("os.path", None),

    # Asyncio operations
    "create_task": ("asyncio", None),
    "gather": ("asyncio", None),
    "sleep": ("asyncio", None),
    "run": ("asyncio", None),
    "get_event_loop": ("asyncio", None),

    # Typing operations
    "List": ("typing", None),
    "Dict": ("typing", None),
    "Set": ("typing", None),
    "Tuple": ("typing", None),
    "Optional": ("typing", None),
    "Union": ("typing", None),
    "Any": ("typing", None),

    # Logging operations
    "getLogger": ("logging", None),

    # JSON operations
    "dumps": ("json", None),
    "loads": ("json", None),

    # Other common operations
    "ArgumentParser": ("argparse", None),
    "defaultdict": ("collections", None),
    "Counter": ("collections", None),
    "deque": ("collections", None),
    "deepcopy": ("copy", None),
    "copy": ("copy", None),
    
    # Additional common functions that are often missing
    "filterwarnings": ("warnings", None),  # Common missing import
    "warn": ("warnings", None),
    "all": None,  # Built-in function
    "any": None,  # Built-in function
    "type": None,  # Built-in function
    "isinstance": None,  # Built-in function
    "hasattr": None,  # Built-in function
    "getattr": None,  # Built-in function
    "setattr": None,  # Built-in function
    "delattr": None,  # Built-in function
    "callable": None,  # Built-in function
    "enumerate": None,  # Built-in function
    "zip": None,  # Built-in function
    "map": None,  # Built-in function
    "filter": None,  # Built-in function
    "sorted": None,  # Built-in function
    "reversed": None,  # Built-in function
    "range": None,  # Built-in function
    "len": None,  # Built-in function
    "str": None,  # Built-in function
    "int": None,  # Built-in function
    "float": None,  # Built-in function
    "bool": None,  # Built-in function
    "list": None,  # Built-in function
    "dict": None,  # Built-in function
    "set": None,  # Built-in function
    "tuple": None,  # Built-in function
    "print": None,  # Built-in function
    "input": None,  # Built-in function
    "open": None,  # Built-in function
    "abs": None,  # Built-in function
    "round": None,  # Built-in function
    "min": None,  # Built-in function
    "max": None,  # Built-in function
    "sum": None,  # Built-in function
}

# Methods that don't need imports (they're attributes of objects)
OBJECT_METHODS = {
    "append", "extend", "insert", "remove", "pop",  # List methods
    "keys", "values", "items", "get", "update",     # Dict methods
    "lower", "upper", "strip", "split", "replace",  # String methods
    "fillna", "rolling", "shift", "diff", "cumsum", # DataFrame methods
    "isoformat", "strftime", "total_seconds",        # DateTime methods
    "exists", "mkdir", "unlink", "rmdir",            # Path methods
}


class ImportFixer:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.issues_by_file = defaultdict(list)
        self.imports_to_add = defaultdict(set)
        self.fixed_files = []
        self.failed_files = []
        
        # Enhanced detection patterns
        self.numpy_patterns = {
            'array', 'zeros', 'ones', 'empty', 'full', 'arange', 'linspace', 'logspace',
            'mean', 'std', 'var', 'sum', 'min', 'max', 'argmin', 'argmax', 'nan', 'inf',
            'isnan', 'isinf', 'isfinite', 'abs', 'sqrt', 'exp', 'log', 'sin', 'cos', 'tan',
            'dot', 'transpose', 'reshape', 'flatten', 'concatenate', 'stack', 'hstack', 'vstack',
            'where', 'clip', 'round'
        }
        
        self.pandas_patterns = {
            'DataFrame', 'Series', 'read_csv', 'read_parquet', 'read_excel', 'read_json',
            'concat', 'merge', 'to_datetime', 'fillna', 'dropna', 'groupby', 'rolling',
            'shift', 'diff', 'cumsum', 'cumprod', 'pivot_table', 'melt', 'get_dummies',
            'cut', 'qcut', 'crosstab', 'value_counts'
        }
        
        self.warnings_patterns = {
            'filterwarnings', 'warn', 'warnings'
        }

    def load_issues(self, json_file: str):
        """Load issues from the validation report."""
        with open(json_file) as f:
            data = json.load(f)

        # Group undefined functions by file
        for issue in data.get("issues", []):
            if issue["issue_type"] == "undefined_function":
                msg = issue["message"]
                if "Function '" in msg:
                    func_name = msg.split("'")[1]
                    if func_name not in OBJECT_METHODS:
                        self.issues_by_file[issue["file_path"]].append({
                            "function": func_name,
                            "line": issue["line_number"],
                        })

    def analyze_imports_needed(self):
        """Analyze which imports are needed for each file."""
        for file_path, issues in self.issues_by_file.items():
            imports_needed = set()

            for issue in issues:
                func = issue["function"]
                if func in COMMON_IMPORTS and COMMON_IMPORTS[func]:
                    module, alias = COMMON_IMPORTS[func]
                    imports_needed.add((module, alias))

            if imports_needed:
                self.imports_to_add[file_path] = imports_needed

    def auto_detect_missing_imports(self, file_path: str) -> set:
        """Automatically detect missing imports by analyzing the file content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            missing_imports = set()
            
            # Find all function calls and attribute access
            for node in ast.walk(tree):
                # Handle direct function calls like np.array()
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    
                    # Check for numpy functions
                    if func_name in self.numpy_patterns:
                        missing_imports.add(('numpy', 'np'))
                    
                    # Check for pandas functions
                    elif func_name in self.pandas_patterns:
                        missing_imports.add(('pandas', 'pd'))
                    
                    # Check for warnings functions
                    elif func_name in self.warnings_patterns:
                        missing_imports.add(('warnings', None))
                
                # Handle attribute access like np.array, pd.DataFrame
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        module_name = node.func.value.id
                        func_name = node.func.attr
                        
                        # Check for numpy patterns
                        if module_name == 'np' and func_name in self.numpy_patterns:
                            missing_imports.add(('numpy', 'np'))
                        
                        # Check for pandas patterns
                        elif module_name == 'pd' and func_name in self.pandas_patterns:
                            missing_imports.add(('pandas', 'pd'))
                
                # Handle attribute access without calls like np.inf
                elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                    module_name = node.value.id
                    attr_name = node.attr
                    
                    # Check for numpy constants
                    if module_name == 'np' and attr_name in {'inf', 'nan', 'pi', 'e'}:
                        missing_imports.add(('numpy', 'np'))
            
            # Check existing imports to avoid duplicates
            existing_imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        existing_imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        existing_imports.add(node.module)
            
            # Filter out already imported modules
            filtered_imports = set()
            for module, alias in missing_imports:
                if module not in existing_imports:
                    filtered_imports.add((module, alias))
            
            return filtered_imports
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return set()

    def auto_fix_file_imports(self, file_path: str) -> bool:
        """Automatically detect and fix missing imports in a file."""
        try:
            # Auto-detect missing imports
            missing_imports = self.auto_detect_missing_imports(file_path)
            
            if not missing_imports:
                return False
            
            # Fix the imports
            return self.fix_file_imports(file_path, missing_imports)
            
        except Exception as e:
            print(f"Error auto-fixing {file_path}: {e}")
            return False

    def fix_file_imports(self, file_path: str, imports_needed: set[tuple[str, str]]) -> bool:
        """Add missing imports to a file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Parse the file to find where to insert imports
            tree = ast.parse(content)

            # Find existing imports at the top of the file only
            existing_imports = set()
            last_import_line = 0
            
            # Only consider imports at the top of the file (before any non-import statements)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        existing_imports.add(alias.name)
                    # Only update last_import_line if this import is at the top
                    if node.lineno <= 50:  # Only consider imports in first 50 lines
                        last_import_line = max(last_import_line, node.lineno)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        existing_imports.add(node.module)
                    # Only update last_import_line if this import is at the top
                    if node.lineno <= 50:  # Only consider imports in first 50 lines
                        last_import_line = max(last_import_line, node.lineno)

            # Prepare new imports
            new_imports = []
            for module, alias in imports_needed:
                if module not in existing_imports:
                    if alias:
                        new_imports.append(f"import {module} as {alias}")
                    else:
                        new_imports.append(f"import {module}")

            if not new_imports:
                return False

            # Insert imports after existing imports or at the beginning
            lines = content.split("\n")

            # Find the right place to insert - be more careful about placement
            if last_import_line > 0:
                # Insert after the last import at the top
                insert_line = last_import_line
            else:
                # No imports found at the top, insert at the beginning
                insert_line = 0
                
                # Handle module docstrings
                if lines and (lines[0].startswith('"""') or lines[0].startswith("'''")):
                    # Find end of docstring
                    for i, line in enumerate(lines[1:], 1):
                        if line.strip().endswith('"""') or line.strip().endswith("'''"):
                            insert_line = i + 1
                            break
                
                # Handle shebang lines
                if lines and lines[0].startswith('#!'):
                    insert_line = 1

            # Insert the imports
            for imp in sorted(new_imports):
                lines.insert(insert_line, imp)
                insert_line += 1

            # Add blank line after imports if needed
            if insert_line < len(lines) and lines[insert_line].strip():
                lines.insert(insert_line, "")

            # Write back
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            return True

        except Exception as e:
            print(f"Error fixing {file_path}: {e}")
            return False

    def generate_report(self) -> dict:
        """Generate a report of fixes to be made."""
        report = {
            "total_files": len(self.imports_to_add),
            "imports_by_file": {},
            "summary": defaultdict(int),
        }

        for file_path, imports in self.imports_to_add.items():
            report["imports_by_file"][file_path] = [
                f"import {module} as {alias}" if alias else f"import {module}"
                for module, alias in imports
            ]

            for module, _ in imports:
                report["summary"][module] += 1

        return report

    def fix_all_imports(self, dry_run: bool = True):
        """Fix imports in all files."""
        self.analyze_imports_needed()

        if dry_run:
            report = self.generate_report()
            print("\nDRY RUN - Imports that would be added:")
            print("=" * 60)

            # Show summary
            print("\nSummary by module:")
            for module, count in sorted(report["summary"].items(), key=lambda x: x[1], reverse=True):
                print(f"  {module}: {count} files")

            # Show sample files
            print("\nSample files to be fixed (showing first 5):")
            for file_path, imports in list(report["imports_by_file"].items())[:5]:
                print(f"\n{file_path}:")
                for imp in imports:
                    print(f"  + {imp}")

            if len(report["imports_by_file"]) > 5:
                print(f"\n... and {len(report['imports_by_file']) - 5} more files")

            return report
        # Actually fix the files
        fixed = 0
        failed = 0

        for file_path, imports in self.imports_to_add.items():
            if self.fix_file_imports(file_path, imports):
                fixed += 1
                print(f"✓ Fixed {file_path}")
            else:
                failed += 1
                print(f"✗ Failed to fix {file_path}")

        print(f"\nFixed {fixed} files, {failed} failures")
        return {"fixed": fixed, "failed": failed}

    def auto_fix_all_files(self, file_paths: list, dry_run: bool = True):
        """Automatically detect and fix missing imports in all files."""
        if dry_run:
            print("\nAUTO-DETECTION DRY RUN - Missing imports that would be added:")
            print("=" * 70)
            
            total_files = 0
            total_imports = 0
            module_counts = defaultdict(int)
            
            for file_path in file_paths:
                missing_imports = self.auto_detect_missing_imports(file_path)
                if missing_imports:
                    total_files += 1
                    total_imports += len(missing_imports)
                    print(f"\n{file_path}:")
                    for module, alias in missing_imports:
                        import_str = f"import {module} as {alias}" if alias else f"import {module}"
                        print(f"  + {import_str}")
                        module_counts[module] += 1
            
            print(f"\nSUMMARY:")
            print(f"  Files with missing imports: {total_files}")
            print(f"  Total imports to add: {total_imports}")
            print(f"\nBy module:")
            for module, count in sorted(module_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {module}: {count} files")
            
            return {"files_to_fix": total_files, "imports_to_add": total_imports, "module_counts": dict(module_counts)}
        
        # Actually fix the files
        fixed = 0
        failed = 0
        
        for file_path in file_paths:
            if self.auto_fix_file_imports(file_path):
                fixed += 1
                self.fixed_files.append(file_path)
                print(f"✓ Auto-fixed {file_path}")
            else:
                failed += 1
                self.failed_files.append(file_path)
        
        print(f"\nAuto-fixed {fixed} files, {failed} failures")
        return {"fixed": fixed, "failed": failed, "fixed_files": self.fixed_files, "failed_files": self.failed_files}


def main():
    import argparse
    import glob

    parser = argparse.ArgumentParser(description="Fix missing imports in Python files")
    parser.add_argument("--project-root", default="/workspace/src",
                       help="Root directory of the project")
    parser.add_argument("--issues-file", default="/workspace/code_quality/interaction_analysis.json",
                       help="JSON file with validation issues")
    parser.add_argument("--fix", action="store_true",
                       help="Actually fix the files (default is dry run)")
    parser.add_argument("--auto-detect", action="store_true",
                       help="Use auto-detection instead of issues file")
    parser.add_argument("--file-pattern", default="**/*.py",
                       help="File pattern for auto-detection (default: **/*.py)")

    args = parser.parse_args()

    fixer = ImportFixer(args.project_root)

    if args.auto_detect:
        # Auto-detect mode: scan all Python files
        print("🔍 Auto-detecting missing imports...")
        file_paths = list(Path(args.project_root).glob(args.file_pattern))
        print(f"Found {len(file_paths)} Python files to analyze")
        
        result = fixer.auto_fix_all_files([str(f) for f in file_paths], dry_run=not args.fix)
        
        # Save report
        if not args.fix:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f"/workspace/code_quality/reports/auto_import_fixes_report_{timestamp}.json"
            with open(report_file, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nReport saved to: {report_file}")
    else:
        # Issues file mode: use existing issues
        fixer.load_issues(args.issues_file)
        result = fixer.fix_all_imports(dry_run=not args.fix)

        # Save report
        if not args.fix:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f"/workspace/code_quality/reports/import_fixes_report_{timestamp}.json"
            with open(report_file, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nReport saved to: {report_file}")


if __name__ == "__main__":
    main()
