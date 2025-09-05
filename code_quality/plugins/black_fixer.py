from typing import Set, List, Dict, Any, Optional
"""
Black code formatter plugin for code quality tools.
"""

import subprocess
import sys
from typing import Any

from .base_plugin import BasePlugin


class BlackFixer(BasePlugin):
    """Black code formatter plugin."""

    def __init__(self, config: dict[str, Any] = None):
        self.name = "Black"
        self.description = "Uncompromising Python code formatter"
        self.version = "1.0.0"
        super().__init__(config)

    def get_name(self) -> str:
        return self.name

    def get_description(self) -> str:
        return self.description

    def get_version(self) -> str:
        return self.version
    
    def get_metadata(self):
        """Return plugin metadata."""
        from plugins.base_plugin import PluginMetadata, PluginCategory, PluginPriority
        return PluginMetadata(
            name=self.name,
            version=self.version,
            description=self.description,
            author="Code Quality Pipeline",
            category=PluginCategory.FORMATTING,
            priority=PluginPriority.HIGH,
            dependencies=["black"],
            tags={"formatting", "black", "code-style"},
            required_packages=["black"]
        )
    
    def is_available(self) -> bool:
        """Check if black is available."""
        try:
            subprocess.run([sys.executable, "-m", "black", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def execute(self, context):
        """Execute the black formatter."""
        from plugins.base_plugin import PluginResult
        if not self.is_available():
            return PluginResult(
                success=False,
                message="black not available",
                data={"skipped": True}
            )
        
        return PluginResult(
            success=True,
            message="Black formatter ready",
            data={"tool": "black"}
        )

    def can_fix(self, file_path: str) -> bool:
        """Check if Black can fix this file."""
        return file_path.endswith(".py")

    def fix(self, file_path: str) -> dict[str, Any]:
        """Fix formatting issues using Black."""
        try:
            # Build Black command
            cmd = [
                sys.executable, "-m", "black",
                "--quiet",  # Suppress output
                "--line-length", str(self.get_config("max_line_length", 88)),
            ]

            # Add aggressive mode if enabled
            if self.get_config("aggressive", False):
                cmd.append("--fast")

            cmd.append(file_path)

            # Run Black
            result = subprocess.run(
                cmd,
                check=False, capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                return {
                    "success": True,
                    "tool": "black",
                    "file": file_path,
                    "message": "Successfully formatted with Black",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            return {
                "success": False,
                "tool": "black",
                "file": file_path,
                "message": f"Black failed with return code {result.returncode}",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "error": True,
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "tool": "black",
                "file": file_path,
                "message": "Black timed out after 60 seconds",
                "error": True,
                "timeout": True,
            }
        except Exception as e:
            return {
                "success": False,
                "tool": "black",
                "file": file_path,
                "message": f"Black failed with exception: {str(e)}",
                "error": True,
                "exception": str(e),
            }

    def get_supported_extensions(self) -> list[str]:
        return [".py"]

    def check_installed(self) -> bool:
        """Check if Black is installed."""
        try:
            subprocess.run(
                [sys.executable, "-m", "black", "--version"],
                capture_output=True,
                check=True,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
