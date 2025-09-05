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
    """Unified pipeline that runs all code quality tools without imports, with plugin support."""

    def __init__(self, project_root: str = "/workspace/src", enable_plugins: bool = True):
        self.project_root = Path(project_root)
        self.code_quality_dir = Path("/workspace/code_quality")
        self.scripts_dir = self.code_quality_dir / "scripts"
        self.reports_dir = self.code_quality_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.enable_plugins = enable_plugins
        
        # Initialize plugin system if enabled
        if self.enable_plugins:
            self._initialize_plugin_system()

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
            "bulk_syntax_cleanup": {
                "script": "bulk_syntax_cleanup.py",
                "description": "Bulk Syntax Cleanup",
                "args": ["--project-root", str(self.project_root)],
            },
            "apply_all_fixes": {
                "script": "apply_all_fixes.py",
                "description": "Apply All Fixes",
                "args": ["--project-root", str(self.project_root)],
            },
            "fix_missing_imports": {
                "script": "fix_missing_imports.py",
                "description": "Fix Missing Imports",
                "args": ["--project-root", str(self.project_root)],
            },
            "add_type_hints": {
                "script": "add_type_hints.py",
                "description": "Add Type Hints",
                "args": ["--project-root", str(self.project_root)],
            },
            "master_code_quality": {
                "script": "master_code_quality.py",
                "description": "Master Code Quality",
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
                "args": ["--project-root", str(self.project_root)],
            },
            "comprehensive_review": {
                "script": "../comprehensive_code_review.py",
                "description": "Comprehensive Code Review",
                "args": ["--project-root", str(self.project_root)],
            },
            "comprehensive_import_undefined_check": {
                "script": "../simple_import_undefined_checker.py",
                "description": "Comprehensive Import and Undefined Checker",
                "args": ["--project-root", str(self.project_root)],
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

    def _initialize_plugin_system(self):
        """Initialize the plugin system for standalone execution."""
        try:
            # Add the code_quality directory to Python path for plugin imports
            import sys
            sys.path.insert(0, str(self.code_quality_dir))
            
            # Import plugin system components
            from plugins.plugin_registry import PluginRegistry
            from plugins.plugin_manager import PluginManager
            
            # Initialize plugin system
            self.plugin_registry = PluginRegistry()
            self.plugin_manager = PluginManager(self.plugin_registry)
            
            # Register available plugins
            self._register_available_plugins()
            
            print("✓ Plugin system initialized successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not initialize plugin system: {e}")
            self.enable_plugins = False

    def _register_available_plugins(self):
        """Register available plugins for standalone execution."""
        try:
            # Register production plugins
            from plugins.production.syntax_fixer import SyntaxFixerPlugin
            from plugins.production.import_fixer import ImportFixerPlugin
            from plugins.production.dead_code_fixer import DeadCodeFixerPlugin
            from plugins.production.linter_runner import LinterRunnerPlugin
            from plugins.production.security_scanner import SecurityScannerPlugin
            
            # Register code quality plugins
            from plugins.black_fixer import BlackFixer
            from plugins.isort_fixer import IsortFixer
            from plugins.autopep8_fixer import Autopep8Fixer
            from plugins.autoflake_fixer import AutoflakeFixer
            from plugins.flake8_analyzer import Flake8Analyzer
            from plugins.ruff_analyzer import RuffAnalyzer
            from plugins.ruff_fixer import RuffFixer
            
            # Register plugins
            self.plugin_registry.register_plugin("syntax_fixer", SyntaxFixerPlugin())
            self.plugin_registry.register_plugin("import_fixer", ImportFixerPlugin())
            self.plugin_registry.register_plugin("dead_code_fixer", DeadCodeFixerPlugin())
            self.plugin_registry.register_plugin("linter_runner", LinterRunnerPlugin())
            self.plugin_registry.register_plugin("security_scanner", SecurityScannerPlugin())
            self.plugin_registry.register_plugin("black_fixer", BlackFixer())
            self.plugin_registry.register_plugin("isort_fixer", IsortFixer())
            self.plugin_registry.register_plugin("autopep8_fixer", Autopep8Fixer())
            self.plugin_registry.register_plugin("autoflake_fixer", AutoflakeFixer())
            self.plugin_registry.register_plugin("flake8_analyzer", Flake8Analyzer())
            self.plugin_registry.register_plugin("ruff_analyzer", RuffAnalyzer())
            self.plugin_registry.register_plugin("ruff_fixer", RuffFixer())
            
            print(f"✓ Registered {len(self.plugin_registry.list_plugins())} plugins")
        except ImportError as e:
            print(f"⚠ Warning: Could not register some plugins: {e}")

    def run_plugin_analysis(self) -> dict[str, Any]:
        """Run plugin-based analysis if plugins are enabled."""
        if not self.enable_plugins:
            return {"status": "disabled", "message": "Plugins are disabled"}

        print("\n" + "="*60)
        print("Running Plugin Analysis")
        print("="*60)

        start_time = time.time()
        results = {}

        # Run each registered plugin
        for plugin_name in self.plugin_registry.list_plugins():
            print(f"Running plugin: {plugin_name}")
            try:
                # Create plugin context
                from plugins.base_plugin import PluginContext
                context = PluginContext(
                    project_root=str(self.project_root),
                    config={},
                    files=list(self.project_root.rglob("*.py"))
                )
                
                # Execute plugin
                plugin_result = self.plugin_manager.execute_plugin(plugin_name, context)
                results[plugin_name] = {
                    "status": "success",
                    "result": plugin_result.to_dict()
                }
            except Exception as e:
                results[plugin_name] = {
                    "status": "error",
                    "error": str(e)
                }

        results["execution_time"] = time.time() - start_time

        # Save individual report
        report_path = self.reports_dir / f"plugin_analysis_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)

        return results

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
            "comprehensive_import_undefined_check": "import_undefined_check_report_*.json",
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
            "syntax_imports": ["syntax_fixer", "import_fixer", "circular_imports", "comprehensive_import_undefined_check"],
            "async_types": ["async_fixer", "type_hints"],
            "consolidated_fixes": ["bulk_syntax_cleanup", "apply_all_fixes", "fix_missing_imports", "add_type_hints", "master_code_quality"],
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

        all_categories = ["syntax_imports", "async_types", "consolidated_fixes", "analysis"]
        categories_to_run = categories or all_categories

        # Run each category
        for category in categories_to_run:
            self.results[category] = self.run_category(category)

        # Run plugin analysis if enabled
        if self.enable_plugins:
            self.results["plugin_analysis"] = self.run_plugin_analysis()

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
                        choices=["syntax_imports", "async_types", "consolidated_fixes", "analysis"],
                        help="Specific categories to run")
    parser.add_argument("--tool", choices=[
                        "syntax_fixer", "import_fixer", "async_fixer",
                        "type_hints", "circular_imports", "function_validator",
                        "comprehensive_review", "comprehensive_import_undefined_check",
                        "bulk_syntax_cleanup", "apply_all_fixes", "fix_missing_imports",
                        "add_type_hints", "master_code_quality",
                        ], help="Run a specific tool only")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Timeout for each tool in seconds")
    parser.add_argument("--no-plugins", action="store_true",
                        help="Disable plugin system")

    args = parser.parse_args()

    pipeline = UnifiedStandalonePipeline(
        project_root=args.project_root,
        enable_plugins=not args.no_plugins
    )

    if args.tool:
        # Run single tool
        result = pipeline.run_tool(args.tool, args.timeout)
        print(json.dumps(result, indent=2))
    else:
        # Run categories or all
        pipeline.run_all(args.categories)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
