#!/usr/bin/env python3
"""
Unified Code Quality Pipeline - Standalone Version with Improved Error Handling

This is a standalone version that doesn't require importing other modules.
It runs all code quality tools using subprocess calls with improved error handling
and reduced redundancy.
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Import base pipeline for common functionality
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipelines.base_pipeline import BasePipeline


class UnifiedStandalonePipeline(BasePipeline):
    """Unified pipeline that runs all code quality tools without imports."""

    def __init__(self, project_root: str = "/workspace/src"):
        super().__init__(project_root)
        self.code_quality_dir = Path("/workspace/code_quality")
        self.scripts_dir = self.code_quality_dir / "scripts"

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

    def run_tool(self, tool_name: str, timeout: int = 300) -> dict[str, Any]:
        """Run a single tool and capture results."""
        if tool_name not in self.tools:
            return self._handle_error(
                Exception(f"Unknown tool: {tool_name}"),
                f"run_tool_{tool_name}"
            )

        tool_info = self.tools[tool_name]
        script_path = self.scripts_dir / tool_info["script"]

        # Check if script exists
        if not script_path.exists():
            return self._handle_error(
                Exception(f"Script not found: {script_path}"),
                f"run_tool_{tool_name}"
            )

        self._print_section_header(f"Running: {tool_info['description']}")

        cmd = ["python3", str(script_path)] + tool_info.get("args", [])

        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                check=False, 
                capture_output=True,
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
            return self._handle_error(
                Exception(f"Timeout after {timeout} seconds"),
                f"run_tool_{tool_name}"
            )
        except Exception as e:
            return self._handle_error(e, f"run_tool_{tool_name}")

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
            return self._handle_error(
                Exception(f"Unknown category: {category}"),
                f"run_category_{category}"
            )

        self._print_section_header(f"Running {category.upper()} Category", 80)

        category_results = {}
        for tool in categories[category]:
            category_results[tool] = self.run_tool(tool)

        return category_results

    def run_all(self, categories: list[str] | None = None) -> dict[str, Any]:
        """Run all tools or specific categories."""
        self._print_pipeline_header("UNIFIED CODE QUALITY PIPELINE - STANDALONE (FIXED)")

        # Validate project root
        if not self._validate_project_root():
            print("Warning: Project root validation failed, but continuing...")

        all_categories = ["syntax_imports", "async_types", "analysis"]
        categories_to_run = categories or all_categories

        # Run each category
        for category in categories_to_run:
            self.results[category] = self.run_category(category)

        # Generate summary
        total_time = self._finalize_execution_tracking()
        summary = self._generate_summary(total_time)
        self.results["summary"] = summary

        # Save comprehensive report
        self._save_report(self.results, "unified_pipeline")

        # Print summary
        self._print_summary(summary)

        return self.results

    def cleanup(self):
        """Cleanup resources used by the pipeline."""
        # Clean up any temporary files or resources
        # This is a placeholder for any cleanup needed
        pass


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run unified code quality pipeline (standalone version with improved error handling)",
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

    with UnifiedStandalonePipeline(args.project_root) as pipeline:
        if args.tool:
            # Run single tool
            result = pipeline.run_tool(args.tool, args.timeout)
            print(json.dumps(result, indent=2))
        else:
            # Run categories or all
            pipeline.run_all(args.categories)


if __name__ == "__main__":
    main()