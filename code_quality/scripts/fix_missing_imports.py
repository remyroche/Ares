#!/usr/bin/env python3
"""
Script to analyze and fix missing imports for common operations.
"""

import ast
import json
from collections import defaultdict
from pathlib import Path

# Common function to module mappings
COMMON_IMPORTS = {
    # DateTime operations
    "now": ("datetime", "datetime"),
    "today": ("datetime", "date"),
    "timedelta": ("datetime", "timedelta"),
    "isoformat": None,  # This is a method, not an import
    "strftime": None,   # This is a method, not an import
    "total_seconds": None,  # This is a method, not an import

    # Pandas operations
    "DataFrame": ("pandas", "pd"),
    "Series": ("pandas", "pd"),
    "read_csv": ("pandas", "pd"),
    "read_parquet": ("pandas", "pd"),
    "concat": ("pandas", "pd"),
    "merge": ("pandas", "pd"),
    "to_datetime": ("pandas", "pd"),

    # NumPy operations
    "array": ("numpy", "np"),
    "zeros": ("numpy", "np"),
    "ones": ("numpy", "np"),
    "mean": ("numpy", "np"),
    "std": ("numpy", "np"),
    "nan": ("numpy", "np"),
    "inf": ("numpy", "np"),
    "isnan": ("numpy", "np"),
    "isinf": ("numpy", "np"),

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

    def fix_file_imports(self, file_path: str, imports_needed: set[tuple[str, str]]) -> bool:
        """Add missing imports to a file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Parse the file to find where to insert imports
            tree = ast.parse(content)

            # Find existing imports
            existing_imports = set()
            last_import_line = 0

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        existing_imports.add(alias.name)
                    last_import_line = max(last_import_line, node.lineno)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        existing_imports.add(node.module)
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

            # Find the right place to insert
            insert_line = max(0, last_import_line)

            # Handle module docstrings
            if insert_line == 0 and lines and (lines[0].startswith('"""') or lines[0].startswith("'''")):
                # Find end of docstring
                for i, line in enumerate(lines[1:], 1):
                    if line.strip().endswith('"""') or line.strip().endswith("'''"):
                        insert_line = i + 1
                        break

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


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fix missing imports in Python files")
    parser.add_argument("--project-root", default="/workspace/src",
                       help="Root directory of the project")
    parser.add_argument("--issues-file", default="/workspace/code_quality/interaction_analysis.json",
                       help="JSON file with validation issues")
    parser.add_argument("--fix", action="store_true",
                       help="Actually fix the files (default is dry run)")

    args = parser.parse_args()

    fixer = ImportFixer(args.project_root)
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
