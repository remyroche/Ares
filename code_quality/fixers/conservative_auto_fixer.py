"""
Conservative Auto-Fixer - A safer version of the auto-fixer that prioritizes not breaking code.
"""

import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.config import CodeQualityConfig, load_config
from ..utils.file_utils import (
    backup_file,
    find_python_files,
    is_valid_python_file,
    restore_file,
)


class ConservativeAutoFixer:
    """
    A conservative auto-fixer that prioritizes safety over aggressive fixes.
    
    Key features:
    - Always validates syntax before and after fixes
    - Creates backups before any modifications
    - Restores files if fixes break syntax
    - Runs only the safest tools by default
    - Checks for pre-existing syntax errors
    """
    
    def __init__(self, config: Optional[CodeQualityConfig] = None):
        """Initialize the conservative auto-fixer."""
        self.config = config or load_config("code_quality/config_conservative.yaml")
        self.fix_results = {}
        self.backup_files = {}
        self.skipped_files = []
        self.validation_errors = {}
        
        # Safety settings
        self.always_backup = True
        self.validate_after_fix = True
        self.restore_on_error = True
        self.skip_broken_files = True
        self.max_file_size_mb = 10
        
        # Conservative tool list (only the safest tools)
        self.safe_tools = ["isort"]  # Start with only isort
        self.moderate_tools = ["autopep8"]  # Requires installation
        self.aggressive_tools = ["black", "yapf"]  # More risky
        
        # Current tool set (can be expanded based on config)
        self.enabled_tools = self._get_enabled_tools()
    
    def _get_enabled_tools(self) -> List[str]:
        """Get list of enabled tools based on configuration."""
        if hasattr(self.config, 'auto_fix') and hasattr(self.config.auto_fix, 'tools'):
            configured_tools = self.config.auto_fix.tools
            # Filter to only include safe tools by default
            return [tool for tool in configured_tools if tool in self.safe_tools]
        return self.safe_tools.copy()
    
    def _validate_syntax(self, file_path: str) -> tuple[bool, Optional[str]]:
        """
        Validate Python syntax for a file.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check file size
            file_size_mb = len(content) / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                return False, f"File too large ({file_size_mb:.1f}MB > {self.max_file_size_mb}MB)"
            
            # Try to parse the AST
            ast.parse(content)
            
            # Try to compile
            compile(content, file_path, 'exec')
            
            return True, None
            
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def _is_tool_available(self, tool: str) -> bool:
        """Check if a tool is available in the system."""
        try:
            if tool in ["black", "isort", "autopep8", "yapf"]:
                # Try to import the module
                result = subprocess.run(
                    [sys.executable, "-m", tool, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                return result.returncode == 0
            return False
        except Exception:
            return False
    
    def _run_isort(self, file_path: str) -> Dict[str, Any]:
        """Run isort with conservative settings."""
        try:
            # Conservative isort command
            cmd = [
                sys.executable, "-m", "isort",
                "--profile", "black",
                "--line-length", "120",
                "--multi-line-output", "3",
                "--trailing-comma",
                "--ensure-newline-before-comments",
                "--honor-noqa",  # Respect # noqa comments
                "--float-to-top", "false",
                "--check-only",  # First, just check
                file_path
            ]
            
            # Check if changes are needed
            check_result = subprocess.run(cmd, capture_output=True, text=True)
            
            if check_result.returncode == 0:
                return {"status": "success", "message": "No changes needed"}
            
            # Apply changes
            cmd.remove("--check-only")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return {"status": "success", "message": "Imports sorted successfully"}
            else:
                return {"status": "failed", "error": result.stderr}
                
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _run_autopep8(self, file_path: str) -> Dict[str, Any]:
        """Run autopep8 with conservative settings."""
        try:
            cmd = [
                sys.executable, "-m", "autopep8",
                "--in-place",
                "--max-line-length", "120",
                "--ignore", "E501,W503,E402,E731",  # Ignore risky fixes
                file_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return {"status": "success", "message": "Code formatted successfully"}
            else:
                return {"status": "failed", "error": result.stderr}
                
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def fix_file(self, file_path: str, tools: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Fix a single file with maximum safety.
        
        Args:
            file_path: Path to the Python file
            tools: List of tools to use (defaults to enabled_tools)
            
        Returns:
            Dictionary with results for each tool
        """
        results = {
            "file": file_path,
            "initial_validation": {},
            "tools": {},
            "final_validation": {},
            "restored": False
        }
        
        # Validate syntax before attempting fixes
        is_valid, error_msg = self._validate_syntax(file_path)
        results["initial_validation"] = {
            "valid": is_valid,
            "error": error_msg
        }
        
        if not is_valid and self.skip_broken_files:
            results["skipped"] = True
            results["skip_reason"] = f"Pre-existing syntax error: {error_msg}"
            self.skipped_files.append(file_path)
            return results
        
        # Create backup
        backup_path = backup_file(file_path)
        self.backup_files[file_path] = backup_path
        results["backup_created"] = backup_path
        
        # Get tools to use
        tools_to_use = tools or self.enabled_tools
        
        # Run each tool
        for tool in tools_to_use:
            if not self._is_tool_available(tool):
                results["tools"][tool] = {
                    "status": "skipped",
                    "reason": "Tool not available"
                }
                continue
            
            # Run the tool
            if tool == "isort":
                tool_result = self._run_isort(file_path)
            elif tool == "autopep8":
                tool_result = self._run_autopep8(file_path)
            else:
                tool_result = {
                    "status": "skipped",
                    "reason": f"Tool {tool} not implemented in conservative mode"
                }
            
            results["tools"][tool] = tool_result
            
            # Validate after each tool
            if self.validate_after_fix and tool_result.get("status") == "success":
                is_valid_after, error_after = self._validate_syntax(file_path)
                
                if not is_valid_after:
                    # Restore from backup
                    if self.restore_on_error:
                        restore_file(backup_path, file_path)
                        results["restored"] = True
                        results["restore_reason"] = f"Tool {tool} broke syntax: {error_after}"
                        results["tools"][tool]["status"] = "rolled_back"
                        break
        
        # Final validation
        final_valid, final_error = self._validate_syntax(file_path)
        results["final_validation"] = {
            "valid": final_valid,
            "error": final_error
        }
        
        # Clean up backup if successful
        if final_valid and not results["restored"]:
            try:
                os.remove(backup_path)
                results["backup_removed"] = True
            except Exception:
                pass
        
        return results
    
    def fix_directory(self, directory: str, recursive: bool = True) -> Dict[str, Any]:
        """
        Fix all Python files in a directory with conservative settings.
        
        Args:
            directory: Directory path
            recursive: Whether to process subdirectories
            
        Returns:
            Dictionary with aggregated results
        """
        results = {
            "directory": directory,
            "total_files": 0,
            "processed_files": 0,
            "skipped_files": 0,
            "restored_files": 0,
            "successful_files": 0,
            "file_results": {},
            "summary": {}
        }
        
        # Find Python files
        pattern = "**/*.py" if recursive else "*.py"
        python_files = []
        
        for file_path in Path(directory).glob(pattern):
            if file_path.is_file() and str(file_path).endswith('.py'):
                # Skip test files and migrations by default
                if any(skip in str(file_path) for skip in ['test_', 'tests/', 'migrations/']):
                    continue
                python_files.append(str(file_path))
        
        results["total_files"] = len(python_files)
        
        print(f"Found {len(python_files)} Python files to process")
        print(f"Using tools: {', '.join(self.enabled_tools)}")
        print("Safety features: Always backup, validate after fix, restore on error")
        print("-" * 50)
        
        # Process each file
        for i, file_path in enumerate(python_files, 1):
            print(f"Processing ({i}/{len(python_files)}): {file_path}")
            
            file_result = self.fix_file(file_path)
            results["file_results"][file_path] = file_result
            
            if file_result.get("skipped"):
                results["skipped_files"] += 1
                print(f"  SKIPPED: {file_result.get('skip_reason', 'Unknown reason')}")
            elif file_result.get("restored"):
                results["restored_files"] += 1
                print(f"  RESTORED: {file_result.get('restore_reason', 'Unknown reason')}")
            elif file_result["final_validation"]["valid"]:
                results["successful_files"] += 1
                print("  SUCCESS: File fixed and validated")
            else:
                print(f"  ERROR: {file_result['final_validation']['error']}")
            
            results["processed_files"] = i
        
        # Generate summary
        results["summary"] = {
            "total_files": results["total_files"],
            "processed_files": results["processed_files"],
            "successful_files": results["successful_files"],
            "skipped_files": results["skipped_files"],
            "restored_files": results["restored_files"],
            "success_rate": (
                results["successful_files"] / results["processed_files"] * 100
                if results["processed_files"] > 0 else 0
            )
        }
        
        print("\n" + "=" * 50)
        print("CONSERVATIVE AUTO-FIX SUMMARY")
        print("=" * 50)
        print(f"Total files found: {results['summary']['total_files']}")
        print(f"Files processed: {results['summary']['processed_files']}")
        print(f"Successfully fixed: {results['summary']['successful_files']}")
        print(f"Skipped (pre-existing errors): {results['summary']['skipped_files']}")
        print(f"Restored (fixes broke syntax): {results['summary']['restored_files']}")
        print(f"Success rate: {results['summary']['success_rate']:.1f}%")
        
        return results


def main():
    """Command-line interface for conservative auto-fixer."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Conservative Python code auto-fixer"
    )
    parser.add_argument(
        "path",
        help="File or directory to fix"
    )
    parser.add_argument(
        "--config",
        help="Configuration file (defaults to conservative config)"
    )
    parser.add_argument(
        "--tools",
        nargs="+",
        choices=["isort", "autopep8", "black", "yapf"],
        help="Specific tools to use"
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't process subdirectories"
    )
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = None  # Will use default conservative config
    
    # Create fixer
    fixer = ConservativeAutoFixer(config)
    
    # Override tools if specified
    if args.tools:
        fixer.enabled_tools = args.tools
    
    # Process path
    path = Path(args.path)
    
    if path.is_file():
        results = fixer.fix_file(str(path))
        print("\nResults:")
        import json
        print(json.dumps(results, indent=2))
    elif path.is_dir():
        results = fixer.fix_directory(
            str(path),
            recursive=not args.no_recursive
        )
    else:
        print(f"Error: {path} is not a valid file or directory")
        return 1
    
    # Return appropriate exit code
    if results.get("summary", {}).get("restored_files", 0) > 0:
        return 2  # Some files had to be restored
    elif results.get("summary", {}).get("successful_files", 0) == 0:
        return 1  # No files were successfully fixed
    else:
        return 0  # Success


if __name__ == "__main__":
    sys.exit(main())