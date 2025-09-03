#!/usr/bin/env python3
"""
Robust async/await fixer that handles complex cases.
"""

import ast
import json
import re
from pathlib import Path


class RobustAsyncFixer:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.fixed_files = []
        self.failed_files = []
        self.async_functions = self._collect_async_functions()

    def _collect_async_functions(self) -> set[str]:
        """Collect names of async functions from the codebase."""
        async_funcs = {
            # Common async functions
            "initialize", "run", "start", "stop", "close", "cleanup",
            "connect", "disconnect", "fetch", "send", "receive",
            "process", "handle", "execute", "wait", "join",
            # Project-specific async functions
            "run_training", "run_validator", "run_step", "test",
            "test_validator", "load_config", "setup_paper_trader",
            "setup_performance_reporter", "setup_enhanced_training_manager",
            "run_full_training", "run_integration_example",
            "create_sample_data", "step1_precompute_features",
            "step2_run_backtests", "step3_performance_comparison",
            "load_unified_data", "run_integrated_pipeline",
            "download_all_data_with_consolidation",
            "setup_sr_detection_optimizer", "setup_sr_optuna_optimizer",
            "setup_regime_specific_optimizer", "start_data_quality_dashboard",
            "run_comprehensive_gap_filling_pipeline", "run_gap_filling_pipeline",
            "get_step_reports", "optimize_dataframe",
            "setup_dual_model_system", "validate_migration_file",
            "export_database_for_trading", "import_database_for_trading",
            "_attempt_recovery", "_execute_pipeline_function",
            "_load_regime_data", "_save_optimization_results",
            "_generate_optimization_report", "_optimize_data_types",
            "_remove_unnecessary_columns", "_optimize_index",
            "_optimize_memory_usage", "main",
        }

        # Also check the issues file for more async functions
        issues_file = Path("/workspace/code_quality/interaction_analysis.json")
        if issues_file.exists():
            try:
                with open(issues_file) as f:
                    data = json.load(f)

                for issue in data.get("issues", []):
                    if issue["issue_type"] == "missing_await":
                        msg = issue["message"]
                        if "Async function '" in msg:
                            func_name = msg.split("'")[1]
                            async_funcs.add(func_name)
            except:
                pass

        return async_funcs

    def fix_missing_awaits(self, file_path: str) -> bool:
        """Fix missing await statements in a file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # First check if file has any async context
            if "async def" not in content and "async with" not in content:
                return False

            lines = content.split("\n")
            modified = False
            in_async_context = False
            context_stack = []

            for i, line in enumerate(lines):
                stripped = line.strip()

                # Track async context
                if "async def" in line:
                    in_async_context = True
                    indent = len(line) - len(line.lstrip())
                    context_stack.append(("async_def", indent))
                elif stripped.startswith("def ") and context_stack:
                    # Check if we're exiting an async context
                    indent = len(line) - len(line.lstrip())
                    while context_stack and context_stack[-1][1] >= indent:
                        context_stack.pop()
                    in_async_context = len(context_stack) > 0

                # Only fix if we're in async context
                if in_async_context and stripped:
                    # Check for async function calls without await
                    for func in self.async_functions:
                        # Various patterns where await might be missing
                        patterns = [
                            (rf"^{func}\s*\(", f"await {func}("),
                            (rf"^(\s*)(.*=\s*){func}\s*\(", r"\1\2await {func}("),
                            (rf"^(\s*)(return\s+){func}\s*\(", r"\1\2await {func}("),
                            (rf"^(\s*)(yield\s+){func}\s*\(", r"\1\2await {func}("),
                        ]

                        for pattern, replacement in patterns:
                            if re.match(pattern, stripped) and "await" not in stripped:
                                # Apply the fix
                                indent = len(line) - len(line.lstrip())
                                new_line = " " * indent + re.sub(pattern, replacement, stripped)
                                lines[i] = new_line
                                modified = True
                                break

                    # Fix asyncio.run(await ...) pattern
                    if "asyncio.run(await" in stripped:
                        lines[i] = lines[i].replace("asyncio.run(await", "asyncio.run(")
                        modified = True

            if modified:
                # Verify the changes don't break syntax
                try:
                    ast.parse("\n".join(lines))
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(lines))
                    self.fixed_files.append(file_path)
                    return True
                except SyntaxError:
                    # Changes broke syntax, don't save
                    pass

            return False

        except Exception:
            return False

    def fix_all_async_issues(self, dry_run: bool = True):
        """Fix async/await issues in all files."""
        # Load the list of files with async issues
        issues_file = Path("/workspace/code_quality/async_fixes_report.json")

        if issues_file.exists():
            with open(issues_file) as f:
                report = json.load(f)

            files_to_fix = list(report.get("issues_by_file", {}).keys())
        else:
            # Scan all Python files
            files_to_fix = list(self.project_root.rglob("*.py"))
            files_to_fix = [
                str(f) for f in files_to_fix
                if "__pycache__" not in str(f) and ".venv" not in str(f)
            ]

        print(f"Checking {len(files_to_fix)} files for async/await issues...")

        if dry_run:
            issues_found = 0
            sample_files = []

            for file_path in files_to_fix[:50]:  # Check first 50 files
                try:
                    with open(file_path) as f:
                        content = f.read()

                    # Quick check for potential issues
                    if "async def" in content:
                        for func in list(self.async_functions)[:10]:
                            if f"{func}(" in content and f"await {func}(" not in content:
                                issues_found += 1
                                sample_files.append(file_path)
                                break
                except:
                    pass

            print(f"\nFound potential async/await issues in {issues_found} files")
            print("\nSample files:")
            for f in sample_files[:5]:
                print(f"  - {Path(f).name}")

            return {"dry_run": True, "potential_fixes": issues_found}
        # Actually fix the files
        for file_path in files_to_fix:
            self.fix_missing_awaits(file_path)

        print(f"\nFixed {len(self.fixed_files)} files")
        print(f"Failed to fix {len(files_to_fix) - len(self.fixed_files)} files")

        return {
            "fixed": len(self.fixed_files),
            "failed": len(files_to_fix) - len(self.fixed_files),
            "fixed_files": self.fixed_files[:10],
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fix async/await issues")
    parser.add_argument("--project-root", default="/workspace/src",
                       help="Root directory of the project")
    parser.add_argument("--fix", action="store_true",
                       help="Actually fix the files (default is dry run)")

    args = parser.parse_args()

    fixer = RobustAsyncFixer(args.project_root)
    result = fixer.fix_all_async_issues(dry_run=not args.fix)

    # Save report
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"/workspace/code_quality/reports/robust_async_fixes_report_{timestamp}.json"
    Path(report_file).parent.mkdir(exist_ok=True)

    with open(report_file, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nReport saved to: {report_file}")


if __name__ == "__main__":
    main()
