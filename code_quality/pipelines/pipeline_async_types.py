#!/usr/bin/env python3
"""
Async and Type Hints Pipeline

This pipeline handles:
1. Async/await fixing
2. Type hint additions and enhancements
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.enhanced_type_hints import TypeHintEnhancer
from scripts.robust_async_fixer import RobustAsyncFixer


class AsyncTypesPipeline:
    """Pipeline for async fixes and type hint enhancements."""

    def __init__(self, project_root: str = "/workspace/src"):
        self.project_root = Path(project_root)
        self.reports_dir = Path("/workspace/code_quality/reports")
        self.reports_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {
            "async_fixes": {},
            "type_hints": {},
            "summary": {},
        }

    def run_async_fixes(self) -> dict[str, Any]:
        """Run robust async/await fixes."""
        print("\n" + "="*60)
        print("Running Async/Await Fixes")
        print("="*60)

        fixer = RobustAsyncFixer(str(self.project_root))
        fixer.fix_all_files()

        result = {
            "fixed_files": fixer.fixed_files,
            "failed_files": fixer.failed_files,
            "total_fixed": len(fixer.fixed_files),
            "total_failed": len(fixer.failed_files),
        }

        # Save report
        report_path = self.reports_dir / f"async_fixes_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)

        self.results["async_fixes"] = result
        return result

    def run_type_hints(self) -> dict[str, Any]:
        """Run type hint enhancements."""
        print("\n" + "="*60)
        print("Running Type Hint Enhancements")
        print("="*60)

        # Get all Python files
        python_files = []
        for pattern in ["**/*.py"]:
            python_files.extend(self.project_root.glob(pattern))

        fixed_files = []
        failed_files = []

        for file_path in python_files:
            try:
                enhancer = TypeHintEnhancer()

                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                # Parse and transform
                tree = ast.parse(content)
                new_tree = enhancer.visit(tree)

                if enhancer.changes_made:
                    # Generate new code
                    new_content = ast.unparse(new_tree)

                    # Add necessary imports
                    if enhancer.imports_needed:
                        import_lines = []
                        if any("Path" in imp for imp in enhancer.imports_needed):
                            import_lines.append("from pathlib import Path")
                        if any("Union" in imp or "Dict" in imp or "List" in imp or "Optional" in imp or "Any" in imp or "Tuple" in imp
                               for imp in enhancer.imports_needed):
                            import_lines.append("from typing import Dict, List, Optional, Union, Any, Tuple")

                        # Insert imports after module docstring and other imports
                        lines = new_content.split("\n")
                        insert_pos = 0
                        for i, line in enumerate(lines):
                            if line.strip() and not line.strip().startswith('"""') and not line.strip().startswith("#"):
                                if line.startswith(("import ", "from ")):
                                    insert_pos = i + 1
                                else:
                                    break

                        for imp in import_lines:
                            lines.insert(insert_pos, imp)
                            insert_pos += 1

                        new_content = "\n".join(lines)

                    # Write back
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(new_content)

                    fixed_files.append({
                        "file": str(file_path),
                        "changes": enhancer.changes_made,
                    })

            except Exception as e:
                failed_files.append({
                    "file": str(file_path),
                    "error": str(e),
                })

        result = {
            "fixed_files": fixed_files,
            "failed_files": failed_files,
            "total_fixed": len(fixed_files),
            "total_failed": len(failed_files),
        }

        # Save report
        report_path = self.reports_dir / f"type_hints_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)

        self.results["type_hints"] = result
        return result

    def run_full_pipeline(self) -> dict[str, Any]:
        """Run the complete async and types pipeline."""
        print("\n" + "="*80)
        print("ASYNC AND TYPE HINTS PIPELINE")
        print("="*80)
        print(f"Project root: {self.project_root}")
        print(f"Timestamp: {self.timestamp}")

        # Run each step
        async_result = self.run_async_fixes()
        type_result = self.run_type_hints()

        # Create summary
        self.results["summary"] = {
            "timestamp": self.timestamp,
            "project_root": str(self.project_root),
            "async_fixes": {
                "fixed": async_result["total_fixed"],
                "failed": async_result["total_failed"],
            },
            "type_hints": {
                "fixed": type_result["total_fixed"],
                "failed": type_result["total_failed"],
            },
        }

        # Save comprehensive report
        report_path = self.reports_dir / f"async_types_pipeline_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2)

        print("\n" + "="*80)
        print("PIPELINE SUMMARY")
        print("="*80)
        print(f"Async fixes: {async_result['total_fixed']} fixed, {async_result['total_failed']} failed")
        print(f"Type hints: {type_result['total_fixed']} fixed, {type_result['total_failed']} failed")
        print(f"\nReports saved to: {self.reports_dir}")

        return self.results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run async fixes and type hints pipeline")
    parser.add_argument("--project-root", default="/workspace/src",
                        help="Project root directory")
    parser.add_argument("--async-only", action="store_true",
                        help="Run only async fixes")
    parser.add_argument("--types-only", action="store_true",
                        help="Run only type hint enhancements")

    args = parser.parse_args()

    pipeline = AsyncTypesPipeline(args.project_root)

    if args.async_only:
        pipeline.run_async_fixes()
    elif args.types_only:
        pipeline.run_type_hints()
    else:
        pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
