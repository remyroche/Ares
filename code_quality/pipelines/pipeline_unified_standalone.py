#!/usr/bin/env python3
"""
Unified Code Quality Pipeline - Standalone Version

This is a standalone version that doesn't require importing other modules.
It runs all code quality tools using subprocess calls.
"""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any


class UnifiedStandalonePipeline:
    """Unified pipeline that runs all code quality tools without imports."""

    def __init__(self, project_root: str = "/workspace/src"):
        self.project_root = Path(project_root)
        self.code_quality_dir = Path("/workspace/code_quality")
        self.scripts_dir = self.code_quality_dir / "scripts"
        self.reports_dir = self.code_quality_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Define all tools and their commands
        self.tools = {
            "syntax_fixer": {
                "script": "advanced_syntax_fixer.py",
                "description": "Advanced Syntax Fixes",
                "args": ["--project-root", str(self.project_root)],
            },
            "import_fixer": {
                "script": "safe_import_fixer.py",
                "description": "Import Fixes",
                "args": ["--project-root", str(self.project_root)],
            },
            "async_fixer": {
                "script": "robust_async_fixer.py",
                "description": "Async/Await Fixes",
                "args": ["--project-root", str(self.project_root)],
            },
            "type_hints": {
                "script": "enhanced_type_hints.py",
                "description": "Type Hint Enhancements",
                "args": ["--project-root", str(self.project_root)],
            },
            "circular_imports": {
                "script": "detect_circular_imports.py",
                "description": "Circular Import Detection",
                "args": ["--project-root", str(self.project_root)],
            },
            "function_validator": {
                "script": "../function_validator.py",
                "description": "Function Validation",
                "args": [str(self.project_root)],
            },
            "comprehensive_review": {
                "script": "../comprehensive_code_review.py",
                "description": "Comprehensive Code Review",
                "args": ["--directory", str(self.project_root)],
            },
        }

        self.results = {}

    def run_tool(self, tool_name: str, timeout: int = 300) -> dict[str, Any]:
        """Run a single tool and capture results."""
        if tool_name not in self.tools:
            return {"error": f"Unknown tool: {tool_name}"}

        tool_info = self.tools[tool_name]
        script_path = self.scripts_dir / tool_info["script"]

        print(f"\n{'='*60}")
        print(f"Running: {tool_info['description']}")
        print(f"{'='*60}")

        cmd = ["python3", str(script_path)] + tool_info.get("args", [])

        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                check=False, capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.code_quality_dir),
            )

            execution_time = time.time() - start_time

            # Try to find and read the generated report
            report_data = self._find_latest_report(tool_name)

            return {
                "tool": tool_name,
                "description": tool_info["description"],
                "success": result.returncode == 0,
                "return_code": result.returncode,
                "execution_time": execution_time,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "report_data": report_data,
            }

        except subprocess.TimeoutExpired:
            return {
                "tool": tool_name,
                "description": tool_info["description"],
                "success": False,
                "error": f"Timeout after {timeout} seconds",
                "execution_time": timeout,
            }
        except Exception as e:
            return {
                "tool": tool_name,
                "description": tool_info["description"],
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time,
            }

    def _find_latest_report(self, tool_name: str) -> dict | None:
        """Find and read the latest report for a tool."""
        # Map tool names to report patterns
        report_patterns = {
            "syntax_fixer": "syntax_fixes_*.json",
            "import_fixer": "import_fixes_*.json",
            "async_fixer": "async_fixes_*.json",
            "type_hints": "type_hints_*.json",
            "circular_imports": "circular_imports_*.json",
            "function_validator": "function_validation_*.json",
            "comprehensive_review": "comprehensive_review_*.json",
        }

        pattern = report_patterns.get(tool_name)
        if not pattern:
            return None

        # Find matching files
        matching_files = list(self.reports_dir.glob(pattern))
        if not matching_files:
            return None

        # Get the most recent file
        latest_file = max(matching_files, key=lambda f: f.stat().st_mtime)

        try:
            with open(latest_file) as f:
                return json.load(f)
        except Exception:
            return None

    def run_category(self, category: str) -> dict[str, Any]:
        """Run all tools in a specific category."""
        categories = {
            "syntax_imports": ["syntax_fixer", "import_fixer", "circular_imports"],
            "async_types": ["async_fixer", "type_hints"],
            "analysis": ["function_validator", "comprehensive_review"],
        }

        if category not in categories:
            return {"error": f"Unknown category: {category}"}

        print(f"\n{'='*80}")
        print(f"Running {category.upper()} Category")
        print(f"{'='*80}")

        category_results = {}
        for tool in categories[category]:
            category_results[tool] = self.run_tool(tool)

        return category_results

    def run_all(self, categories: list[str] | None = None) -> dict[str, Any]:
        """Run all tools or specific categories."""
        print(f"\n{'='*80}")
        print("UNIFIED CODE QUALITY PIPELINE - STANDALONE")
        print(f"{'='*80}")
        print(f"Project root: {self.project_root}")
        print(f"Timestamp: {self.timestamp}")

        all_categories = ["syntax_imports", "async_types", "analysis"]
        categories_to_run = categories or all_categories

        # Run each category
        for category in categories_to_run:
            self.results[category] = self.run_category(category)

        # Generate summary
        summary = self._generate_summary()
        self.results["summary"] = summary

        # Save comprehensive report
        report_path = self.reports_dir / f"unified_pipeline_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2)

        # Print summary
        self._print_summary(summary)

        return self.results

    def _generate_summary(self) -> dict[str, Any]:
        """Generate summary of all results."""
        summary = {
            "timestamp": self.timestamp,
            "project_root": str(self.project_root),
            "categories_run": list(self.results.keys()),
            "tools_summary": {},
            "total_execution_time": 0,
            "successful_tools": 0,
            "failed_tools": 0,
        }

        for category, tools in self.results.items():
            if category == "summary":
                continue

            for tool_name, result in tools.items():
                summary["tools_summary"][tool_name] = {
                    "success": result.get("success", False),
                    "execution_time": result.get("execution_time", 0),
                    "error": result.get("error"),
                }

                summary["total_execution_time"] += result.get("execution_time", 0)
                if result.get("success"):
                    summary["successful_tools"] += 1
                else:
                    summary["failed_tools"] += 1

        return summary

    def _print_summary(self, summary: dict[str, Any]):
        """Print a formatted summary."""
        print(f"\n{'='*80}")
        print("PIPELINE SUMMARY")
        print(f"{'='*80}")
        print(f"Total execution time: {summary['total_execution_time']:.2f} seconds")
        print(f"Successful tools: {summary['successful_tools']}")
        print(f"Failed tools: {summary['failed_tools']}")
        print("\nTool Results:")

        for tool, info in summary["tools_summary"].items():
            status = "✓" if info["success"] else "✗"
            time_str = f"{info['execution_time']:.2f}s"
            error_str = f" - {info['error']}" if info.get("error") else ""
            print(f"  {status} {tool}: {time_str}{error_str}")

        print(f"\nReports saved to: {self.reports_dir}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run unified code quality pipeline (standalone version)",
    )
    parser.add_argument("--project-root", default="/workspace/src",
                        help="Project root directory")
    parser.add_argument("--categories", nargs="+",
                        choices=["syntax_imports", "async_types", "analysis"],
                        help="Specific categories to run")
    parser.add_argument("--tool", choices=[
                        "syntax_fixer", "import_fixer", "async_fixer",
                        "type_hints", "circular_imports", "function_validator",
                        "comprehensive_review",
                        ], help="Run a specific tool only")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Timeout for each tool in seconds")

    args = parser.parse_args()

    pipeline = UnifiedStandalonePipeline(args.project_root)

    if args.tool:
        # Run single tool
        result = pipeline.run_tool(args.tool, args.timeout)
        print(json.dumps(result, indent=2))
    else:
        # Run categories or all
        pipeline.run_all(args.categories)


if __name__ == "__main__":
    main()
