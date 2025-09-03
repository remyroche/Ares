"""
Main auto-fixer module that orchestrates all code fixing operations.
"""

import configparser
import os
import subprocess
import sys
import tempfile
from typing import Any

try:
    import tomllib as toml  # Python 3.11+
except Exception:  # pragma: no cover
    toml = None

from ..core.config import CodeQualityConfig, get_default_config
from ..core.plugins import PluginManager
from ..utils.file_utils import (
    backup_file,
    find_python_files,
    is_valid_python_file,
    restore_file,
)
from ..utils.progress import ProgressManager


class AutoFixer:
    """
    Main class for automatically fixing Python code issues.
    """

    def __init__(self, config: CodeQualityConfig | None = None):
        self.config = config or get_default_config()
        self.fix_results = {}
        self.backup_files = {}

        # Initialize plugin manager and progress manager
        self.plugin_manager = PluginManager(self.config.__dict__)
        self.progress_manager = ProgressManager()

        # Try to unify tool configs from pyproject/setup.cfg
        self._unify_tool_configurations()

        # Register built-in plugins
        self._register_builtin_plugins()

    def fix_all(self, directory: str) -> dict[str, Any]:
        """
        Fix all issues in a directory using configured tools.

        Args:
            directory: Directory containing Python files to fix

        Returns:
            Dictionary containing fix results
        """
        if not self.config.auto_fix.enabled:
            print("Auto-fixing is disabled in configuration.")
            return {}

        python_files = find_python_files(directory, self.config.analysis.exclude_patterns)
        print(f"Found {len(python_files)} Python files to process.")

        # Create backups
        self._create_backups(python_files)

        try:
            # Use progress manager to track the fixing operation with simple concurrency
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def run_for_file(file_path: str) -> dict[str, Any]:
                return self._fix_single_file(file_path)

            results = {}
            max_workers = min(8, max(1, os.cpu_count() or 2))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {executor.submit(run_for_file, f): f for f in python_files}
                for future in as_completed(future_map):
                    file_path = future_map[future]
                    try:
                        results[file_path] = future.result()
                    except Exception as e:
                        results[file_path] = {"success": False, "error": str(e)}

            self.fix_results = results

            # Validate fixes
            self._validate_fixes(python_files)

        except Exception as e:
            print(f"Error during fixing: {e}")
            self._restore_backups()
            raise

        return self.fix_results

    def _unify_tool_configurations(self) -> None:
        """Best-effort read of project config to set line length and options."""
        try:
            project_root = Path(self.config.project_root or os.getcwd())
            pyproject = project_root / "pyproject.toml"
            setup_cfg = project_root / "setup.cfg"

            line_length = None

            if toml and pyproject.exists():
                try:
                    with open(pyproject, "rb") as f:
                        data = toml.load(f)
                    # Look for black/ruff settings
                    black = data.get("tool", {}).get("black", {})
                    ruff = data.get("tool", {}).get("ruff", {})
                    if isinstance(black, dict) and "line-length" in black:
                        line_length = int(black["line-length"])  # pyproject uses hyphen
                    elif isinstance(ruff, dict):
                        # ruff can nest format options
                        fmt = ruff.get("format", {})
                        if isinstance(fmt, dict) and "line-length" in fmt:
                            line_length = int(fmt["line-length"])  # type: ignore[arg-type]
                except Exception:
                    pass

            if line_length is None and setup_cfg.exists():
                try:
                    parser = configparser.ConfigParser()
                    parser.read(setup_cfg)
                    if parser.has_section("flake8") and parser.has_option("flake8", "max-line-length"):
                        line_length = parser.getint("flake8", "max-line-length")
                except Exception:
                    pass

            if line_length and line_length != self.config.auto_fix.max_line_length:
                self.config.auto_fix.max_line_length = int(line_length)
        except Exception:
            # Non-fatal
            pass

    def _register_builtin_plugins(self):
        """Register built-in code fixing plugins."""
        try:
            from code_quality.plugins.autoflake_fixer import AutoflakeFixer
            from code_quality.plugins.autopep8_fixer import (
                Autopep8Fixer as Autopep8Fixer_code_quality_plugins_autopep8_fixer,
            )
            from code_quality.plugins.black_fixer import BlackFixer
            from code_quality.plugins.docformatter_fixer import DocformatterFixer as DocformatterFixer_2
            from code_quality.plugins.flynt_fixer import FlyntFixer
            from code_quality.plugins.future_annotations_fixer import FutureAnnotationsFixer
            from code_quality.plugins.import_hygiene_fixer import ImportHygieneFixer
            from code_quality.plugins.isort_fixer import IsortFixer
            from code_quality.plugins.pyupgrade_fixer import PyupgradeFixer
            from code_quality.plugins.ruff_fixer import RuffFixer
            from code_quality.plugins.unify_fixer import UnifyFixer
            from code_quality.plugins.yapf_fixer import YapfFixer
            from code_quality.plugins.yesqa_fixer import YesqaFixer

            # Register plugins with configuration
            black_config = {
                "enabled": True,
                "max_line_length": self.config.auto_fix.max_line_length,
                "aggressive": self.config.auto_fix.aggressive,
            }

            isort_config = {
                "enabled": True,
                "max_line_length": self.config.auto_fix.max_line_length,
                "aggressive": self.config.auto_fix.aggressive,
            }
            autopep8_config = {
                "enabled": True,
                "max_line_length": self.config.auto_fix.max_line_length,
                "aggressive": self.config.auto_fix.aggressive,
            }
            yapf_config = {
                "enabled": True,
                "max_line_length": self.config.auto_fix.max_line_length,
            }
            docformatter_config = {
                "enabled": True,
                "max_line_length": self.config.auto_fix.max_line_length,
            }
            unify_config = {
                "enabled": True,
            }

            # Only register tools present in config.auto_fix.tools
            tools = set(self.config.auto_fix.tools or [])
            if not tools:
                tools = {"black", "isort"}

            if "black" in tools:
                self.plugin_manager.register_plugin("black", BlackFixer(black_config))
            if "isort" in tools:
                self.plugin_manager.register_plugin("isort", IsortFixer(isort_config))
            if "autopep8" in tools:
                self.plugin_manager.register_plugin("autopep8", Autopep8Fixer(autopep8_config))
            if "yapf" in tools:
                self.plugin_manager.register_plugin("yapf", YapfFixer(yapf_config))
            if "docformatter" in tools:
                self.plugin_manager.register_plugin("docformatter", DocformatterFixer(docformatter_config))
            if "unify" in tools:
                self.plugin_manager.register_plugin("unify", UnifyFixer(unify_config))
            if "ruff" in tools:
                self.plugin_manager.register_plugin("ruff", RuffFixer({"max_line_length": self.config.auto_fix.max_line_length}))
            if "pyupgrade" in tools:
                self.plugin_manager.register_plugin("pyupgrade", PyupgradeFixer({"py311_plus": True}))
            if "flynt" in tools:
                self.plugin_manager.register_plugin("flynt", FlyntFixer({"aggressive": self.config.auto_fix.aggressive}))
            if "autoflake" in tools:
                self.plugin_manager.register_plugin("autoflake", AutoflakeFixer({}))
            if "yesqa" in tools:
                self.plugin_manager.register_plugin("yesqa", YesqaFixer({}))
            if "import_hygiene" in tools:
                self.plugin_manager.register_plugin("import_hygiene", ImportHygieneFixer({}))
            if "future_annotations" in tools:
                self.plugin_manager.register_plugin("future_annotations", FutureAnnotationsFixer({"enabled": True}))

        except ImportError as e:
            print(f"Warning: Could not import built-in plugins: {e}")

    def _fix_single_file(self, file_path: str) -> dict[str, Any]:
        """Fix a single file using available plugins."""
        try:
            available_fixers = self.plugin_manager.get_available_fixers(file_path)

            if not available_fixers:
                return {
                    "success": False,
                    "message": "No suitable fixers available",
                    "fixers_used": [],
                }

            file_results = []
            for fixer in available_fixers:
                try:
                    result = fixer.fix(file_path)
                    file_results.append(result)
                except Exception as e:
                    file_results.append({
                        "success": False,
                        "tool": fixer.get_name(),
                        "error": str(e),
                    })

            return {
                "success": any(r.get("success", False) for r in file_results),
                "fixers_used": [f.get_name() for f in available_fixers],
                "results": file_results,
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Error fixing file: {str(e)}",
                "error": str(e),
            }

    def fix_file(self, file_path: str) -> dict[str, Any]:
        """
        Fix a single Python file using configured tools.

        Args:
            file_path: Path to the Python file to fix

        Returns:
            Dictionary containing fix results
        """
        if not self.config.auto_fix.enabled:
            print("Auto-fixing is disabled in configuration.")
            return {}

        if not file_path.endswith(".py"):
            print(f"Warning: {file_path} is not a Python file.")
            return {}

        print(f"Fixing single file: {file_path}")

        # Create backup
        self._create_backups([file_path])

        try:
            # Run all configured fixers
            for tool in self.config.auto_fix.tools:
                if tool == "black":
                    self._run_black([file_path])
                elif tool == "isort":
                    self._run_isort([file_path])
                elif tool == "autopep8":
                    self._run_autopep8([file_path])
                elif tool == "yapf":
                    self._run_yapf([file_path])
                elif tool == "ruff":
                    self._run_ruff([file_path])
                else:
                    print(f"Warning: Unknown tool '{tool}' configured.")

            # Validate fixes
            self._validate_fixes([file_path])

        except Exception as e:
            print(f"Error during fixing: {e}")
            self._restore_backups()
            raise

        return self.fix_results

    def _create_backups(self, files: list[str]) -> None:
        """Create backups of all files before fixing."""
        print("Creating backups...")
        for file_path in files:
            try:
                backup_path = backup_file(file_path)
                self.backup_files[file_path] = backup_path
            except Exception as e:
                print(f"Warning: Could not backup {file_path}: {e}")

    def _restore_backups(self) -> None:
        """Restore all files from backups."""
        print("Restoring from backups...")
        for original_path, backup_path in self.backup_files.items():
            try:
                restore_file(backup_path, original_path)
                os.remove(backup_path)
            except Exception as e:
                print(f"Warning: Could not restore {original_path}: {e}")

    def _run_black(self, files: list[str]) -> None:
        """Run Black code formatter."""
        print("Running Black formatter...")
        try:
            cmd = [
                sys.executable, "-m", "black",
                "--line-length", str(self.config.auto_fix.max_line_length),
                "--quiet",
            ]

            if self.config.auto_fix.aggressive:
                cmd.append("--fast")

            cmd.extend(files)

            result = subprocess.run(cmd, check=False, capture_output=True, text=True)

            if result.returncode == 0:
                print("Black formatting completed successfully.")
                self.fix_results["black"] = {"status": "success", "files_processed": len(files)}
            else:
                print(f"Black formatting failed: {result.stderr}")
                self.fix_results["black"] = {"status": "failed", "error": result.stderr}

        except Exception as e:
            print(f"Error running Black: {e}")
            self.fix_results["black"] = {"status": "error", "error": str(e)}

    def _run_isort(self, files: list[str]) -> None:
        """Run isort import organizer."""
        print("Running isort...")
        try:
            cmd = [
                sys.executable, "-m", "isort",
                "--profile", "black",
                "--line-length", str(self.config.auto_fix.max_line_length),
                "--quiet",
            ]

            cmd.extend(files)

            result = subprocess.run(cmd, check=False, capture_output=True, text=True)

            if result.returncode == 0:
                print("isort completed successfully.")
                self.fix_results["isort"] = {"status": "success", "files_processed": len(files)}
            else:
                print(f"isort failed: {result.stderr}")
                self.fix_results["isort"] = {"status": "failed", "error": result.stderr}

        except Exception as e:
            print(f"Error running isort: {e}")
            self.fix_results["isort"] = {"status": "error", "error": str(e)}

    def _run_autopep8(self, files: list[str]) -> None:
        """Run autopep8 code formatter."""
        print("Running autopep8...")
        try:
            cmd = [
                sys.executable, "-m", "autopep8",
                "--max-line-length", str(self.config.auto_fix.max_line_length),
                "--aggressive" if self.config.auto_fix.aggressive else "--in-place",
                "--recursive",
            ]

            cmd.extend(files)

            result = subprocess.run(cmd, check=False, capture_output=True, text=True)

            if result.returncode == 0:
                print("autopep8 completed successfully.")
                self.fix_results["autopep8"] = {"status": "success", "files_processed": len(files)}
            else:
                print(f"autopep8 failed: {result.stderr}")
                self.fix_results["autopep8"] = {"status": "failed", "error": result.stderr}

        except Exception as e:
            print(f"Error running autopep8: {e}")
            self.fix_results["autopep8"] = {"status": "error", "error": str(e)}

    def _run_yapf(self, files: list[str]) -> None:
        """Run yapf code formatter."""
        print("Running yapf...")
        try:
            # Create temporary style configuration
            style_config = f"""
[style]
COLUMN_LIMIT = {self.config.auto_fix.max_line_length}
INDENT_WIDTH = 4
USE_TABS = False
"""

            with tempfile.NamedTemporaryFile(mode="w", suffix=".style", delete=False) as f:
                f.write(style_config)
                style_file = f.name

            try:
                cmd = [
                    sys.executable, "-m", "yapf",
                    "--style", style_file,
                    "--in-place",
                ]

                cmd.extend(files)

                result = subprocess.run(cmd, check=False, capture_output=True, text=True)

                if result.returncode == 0:
                    print("yapf completed successfully.")
                    self.fix_results["yapf"] = {"status": "success", "files_processed": len(files)}
                else:
                    print(f"yapf failed: {result.stderr}")
                    self.fix_results["yapf"] = {"status": "failed", "error": result.stderr}

            finally:
                os.unlink(style_file)

        except Exception as e:
            print(f"Error running yapf: {e}")
            self.fix_results["yapf"] = {"status": "error", "error": str(e)}

    def _run_ruff(self, files: list[str]) -> None:
        """Run Ruff linter and formatter."""
        print("Running Ruff...")
        try:
            # First, check if ruff is available
            try:
                subprocess.run([sys.executable, "-m", "ruff", "--version"],
                             capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("Warning: Ruff not available, skipping...")
                self.fix_results["ruff"] = {"status": "skipped", "error": "Ruff not installed"}
                return

            # Run ruff format
            format_cmd = [
                sys.executable, "-m", "ruff", "format",
                "--line-length", str(self.config.auto_fix.max_line_length),
            ]

            format_cmd.extend(files)

            format_result = subprocess.run(format_cmd, check=False, capture_output=True, text=True)

            if format_result.returncode == 0:
                print("Ruff formatting completed successfully.")
                format_status = "success"
            else:
                print(f"Ruff formatting failed: {format_result.stderr}")
                format_status = "failed"

            # Run ruff check and auto-fix
            check_cmd = [
                sys.executable, "-m", "ruff", "check",
                "--fix",
                "--line-length", str(self.config.auto_fix.max_line_length),
            ]

            check_cmd.extend(files)

            check_result = subprocess.run(check_cmd, check=False, capture_output=True, text=True)

            if check_result.returncode in [0, 1]:  # ruff returns 1 when issues are found and fixed
                print("Ruff checking and auto-fixing completed successfully.")
                check_status = "success"
            else:
                print(f"Ruff checking failed: {check_result.stderr}")
                check_status = "failed"

            # Overall ruff status
            if format_status == "success" and check_status == "success":
                overall_status = "success"
            elif format_status == "failed" or check_status == "failed":
                overall_status = "failed"
            else:
                overall_status = "partial"

            self.fix_results["ruff"] = {
                "status": overall_status,
                "files_processed": len(files),
                "format_status": format_status,
                "check_status": check_status,
                "format_output": format_result.stdout,
                "check_output": check_result.stdout,
            }

        except Exception as e:
            print(f"Error running Ruff: {e}")
            self.fix_results["ruff"] = {"status": "error", "error": str(e)}

    def _validate_fixes(self, files: list[str]) -> None:
        """Validate that fixes didn't break syntax."""
        print("Validating fixes...")
        invalid_files = []

        for file_path in files:
            if not is_valid_python_file(file_path):
                invalid_files.append(file_path)

        if invalid_files:
            print(f"Warning: {len(invalid_files)} files have syntax errors after fixing:")
            for file_path in invalid_files:
                print(f"  - {file_path}")

            # Restore backups for invalid files
            self._restore_invalid_files(invalid_files)
        else:
            print("All files have valid syntax after fixing.")
            # Clean up backups
            self._cleanup_backups()

    def _restore_invalid_files(self, invalid_files: list[str]) -> None:
        """Restore specific files from backups."""
        for file_path in invalid_files:
            if file_path in self.backup_files:
                try:
                    restore_file(self.backup_files[file_path], file_path)
                    print(f"Restored {file_path} from backup.")
                except Exception as e:
                    print(f"Warning: Could not restore {file_path}: {e}")

    def _cleanup_backups(self) -> None:
        """Clean up backup files."""
        for backup_path in self.backup_files.values():
            try:
                os.remove(backup_path)
            except Exception as e:
                print(f"Warning: Could not remove backup {backup_path}: {e}")

        self.backup_files.clear()

    def get_fix_summary(self) -> dict[str, Any]:
        """Get a summary of all fix operations."""
        return {
            "total_tools": len(self.config.auto_fix.tools),
            "tools_run": list(self.fix_results.keys()),
            "successful_tools": [tool for tool, result in self.fix_results.items()
                               if result.get("status") == "success"],
            "failed_tools": [tool for tool, result in self.fix_results.items()
                           if result.get("status") in ["failed", "error"]],
            "details": self.fix_results,
        }



def main():
    """Command-line interface for the auto-fixer."""
    import argparse

    parser = argparse.ArgumentParser(description="Auto-fix Python code issues")
    parser.add_argument("--path", required=True, help="Path to directory or file containing Python code")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--max-line-length", type=int, default=88, help="Maximum line length")
    parser.add_argument("--aggressive", action="store_true", help="Enable aggressive fixing")

    args = parser.parse_args()

    # Load configuration
    if args.config:
        from ..core.config import load_config
        config = load_config(args.config)
    else:
        config = get_default_config()

    # Update config with command line arguments
    config.auto_fix.max_line_length = args.max_line_length
    config.auto_fix.aggressive = args.aggressive

    # Run auto-fixer
    fixer = AutoFixer(config)

    # Check if path is a file or directory
    if os.path.isfile(args.path):
        fixer.fix_file(args.path)
    else:
        fixer.fix_all(args.path)

    # Print summary
    summary = fixer.get_fix_summary()
    print("\n" + "="*50)
    print("AUTO-FIX SUMMARY")
    print("="*50)
    print(f"Tools configured: {summary['total_tools']}")
    print(f"Tools run: {', '.join(summary['tools_run'])}")
    print(f"Successful: {', '.join(summary['successful_tools'])}")
    print(f"Failed: {', '.join(summary['failed_tools'])}")

    if summary["failed_tools"]:
        print("\nFailed tool details:")
        for tool in summary["failed_tools"]:
            error = summary["details"][tool].get("error", "Unknown error")
            print(f"  {tool}: {error}")


if __name__ == "__main__":
    main()
