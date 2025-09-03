#!/usr/bin/env python3
"""
Script to fix missing await statements for async function calls.
"""

import ast
import json
import re
from collections import defaultdict
from pathlib import Path


class AsyncAwaitFixer(ast.NodeTransformer):
    """AST transformer to add missing await statements."""

    def __init__(self, async_functions: set[str]):
        self.async_functions = async_functions
        self.in_async_function = False
        self.changes_made = []

    def visit_AsyncFunctionDef(self, node):
        """Track when we're inside an async function."""
        old_state = self.in_async_function
        self.in_async_function = True
        self.generic_visit(node)
        self.in_async_function = old_state
        return node

    def visit_FunctionDef(self, node):
        """Track when we're inside a regular function."""
        old_state = self.in_async_function
        self.in_async_function = False
        self.generic_visit(node)
        self.in_async_function = old_state
        return node

    def visit_Call(self, node):
        """Check if async function calls need await."""
        self.generic_visit(node)

        if self.in_async_function:
            func_name = self._get_function_name(node)
            if func_name in self.async_functions:
                # Check if it's already awaited
                if not self._is_awaited(node):
                    # Create an await expression
                    await_node = ast.Await(value=node)
                    self.changes_made.append(f"Added await to {func_name}")
                    return ast.copy_location(await_node, node)

        return node

    def _get_function_name(self, node):
        """Extract function name from a Call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None

    def _is_awaited(self, node):
        """Check if a node is already wrapped in an await."""
        # This is a simplified check - in practice, we'd need to check the parent
        return False


class AsyncPatternFixer:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.async_issues = defaultdict(list)
        self.async_functions = set()

    def load_issues(self, json_file: str):
        """Load async/await issues from the validation report."""
        with open(json_file) as f:
            data = json.load(f)

        for issue in data.get("issues", []):
            if issue["issue_type"] == "missing_await":
                msg = issue["message"]
                if "Async function '" in msg:
                    func_name = msg.split("'")[1]
                    self.async_functions.add(func_name)
                    self.async_issues[issue["file_path"]].append({
                        "function": func_name,
                        "line": issue["line_number"],
                        "message": msg,
                    })

    def fix_file_async(self, file_path: str, issues: list[dict]) -> bool:
        """Fix async/await issues in a file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Parse the file
            tree = ast.parse(content)

            # Apply fixes
            fixer = AsyncAwaitFixer(self.async_functions)
            new_tree = fixer.visit(tree)

            if fixer.changes_made:
                # Generate new code
                new_code = ast.unparse(new_tree)

                # Write back
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_code)

                return True

            return False

        except Exception as e:
            print(f"Error fixing {file_path}: {e}")
            return False

    def fix_with_regex(self, file_path: str, issues: list[dict]) -> bool:
        """Alternative approach using regex for simpler cases."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            lines = content.split("\n")
            modified = False

            for issue in issues:
                line_num = issue["line"] - 1  # Convert to 0-based
                if 0 <= line_num < len(lines):
                    line = lines[line_num]
                    func_name = issue["function"]

                    # Simple pattern matching for common cases
                    patterns = [
                        (rf"(\s*)({func_name}\s*\([^)]*\))", r"\1await \2"),
                        (rf"(\s*)result\s*=\s*({func_name}\s*\([^)]*\))", r"\1result = await \2"),
                        (rf"(\s*)data\s*=\s*({func_name}\s*\([^)]*\))", r"\1data = await \2"),
                    ]

                    for pattern, replacement in patterns:
                        if re.search(pattern, line) and "await" not in line:
                            new_line = re.sub(pattern, replacement, line)
                            lines[line_num] = new_line
                            modified = True
                            break

            if modified:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))
                return True

            return False

        except Exception as e:
            print(f"Error fixing {file_path}: {e}")
            return False

    def generate_report(self) -> dict:
        """Generate a report of async/await fixes needed."""
        report = {
            "total_files": len(self.async_issues),
            "total_issues": sum(len(issues) for issues in self.async_issues.values()),
            "issues_by_file": {},
            "functions_needing_await": list(self.async_functions),
        }

        for file_path, issues in self.async_issues.items():
            report["issues_by_file"][file_path] = [
                f"Line {issue['line']}: {issue['function']}() needs await"
                for issue in issues
            ]

        return report

    def fix_all_async(self, dry_run: bool = True):
        """Fix all async/await issues."""
        if dry_run:
            report = self.generate_report()
            print("\nDRY RUN - Async/await fixes needed:")
            print("=" * 60)

            print(f"\nTotal files with issues: {report['total_files']}")
            print(f"Total missing await statements: {report['total_issues']}")

            print("\nAsync functions that need await:")
            for func in sorted(report["functions_needing_await"][:10]):
                print(f"  - {func}()")

            if len(report["functions_needing_await"]) > 10:
                print(f"  ... and {len(report['functions_needing_await']) - 10} more")

            print("\nSample files to be fixed (showing first 5):")
            for file_path, issues in list(report["issues_by_file"].items())[:5]:
                print(f"\n{file_path}:")
                for issue in issues[:3]:
                    print(f"  - {issue}")
                if len(issues) > 3:
                    print(f"  ... and {len(issues) - 3} more")

            return report
        # Actually fix the files
        fixed = 0
        failed = 0

        for file_path, issues in self.async_issues.items():
            # Try regex approach first (simpler and preserves formatting)
            if self.fix_with_regex(file_path, issues):
                fixed += 1
                print(f"✓ Fixed {file_path}")
            else:
                failed += 1
                print(f"✗ Failed to fix {file_path}")

        print(f"\nFixed {fixed} files, {failed} failures")
        return {"fixed": fixed, "failed": failed}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fix missing await statements")
    parser.add_argument("--project-root", default="/workspace/src",
                       help="Root directory of the project")
    parser.add_argument("--issues-file", default="/workspace/code_quality/interaction_analysis.json",
                       help="JSON file with validation issues")
    parser.add_argument("--fix", action="store_true",
                       help="Actually fix the files (default is dry run)")

    args = parser.parse_args()

    fixer = AsyncPatternFixer(args.project_root)
    fixer.load_issues(args.issues_file)

    result = fixer.fix_all_async(dry_run=not args.fix)

    # Save report
    if not args.fix:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"/workspace/code_quality/reports/async_fixes_report_{timestamp}.json"
        with open(report_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nReport saved to: {report_file}")


if __name__ == "__main__":
    main()
