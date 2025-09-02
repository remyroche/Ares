"""
docformatter plugin for formatting docstrings.
"""

import subprocess
import sys
from typing import Dict, Any, List

from code_quality.core.plugins import BaseCodeFixer


class DocformatterFixer(BaseCodeFixer):
    """docformatter plugin to format Python docstrings consistently."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "docformatter"
        self.description = "Formats docstrings to follow PEP 257"
        self.version = "1.0.0"

    def get_name(self) -> str:
        return self.name

    def get_description(self) -> str:
        return self.description

    def get_version(self) -> str:
        return self.version

    def can_fix(self, file_path: str) -> bool:
        return file_path.endswith('.py')

    def fix(self, file_path: str) -> Dict[str, Any]:
        """Format docstrings using docformatter."""
        try:
            max_len = str(self.get_config('max_line_length', 88))
            cmd: List[str] = [
                sys.executable, "-m", "docformatter",
                "-i",
                "--wrap-summaries", max_len,
                "--wrap-descriptions", max_len,
                file_path
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                return {
                    'success': True,
                    'tool': 'docformatter',
                    'file': file_path,
                    'message': 'Successfully formatted docstrings with docformatter',
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
            else:
                return {
                    'success': False,
                    'tool': 'docformatter',
                    'file': file_path,
                    'message': f'docformatter failed with return code {result.returncode}',
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'error': True
                }

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'tool': 'docformatter',
                'file': file_path,
                'message': 'docformatter timed out after 60 seconds',
                'error': True,
                'timeout': True
            }
        except Exception as e:
            return {
                'success': False,
                'tool': 'docformatter',
                'file': file_path,
                'message': f'docformatter failed with exception: {str(e)}',
                'error': True,
                'exception': str(e)
            }

    def get_supported_extensions(self) -> List[str]:
        return ['.py']

    def check_installed(self) -> bool:
        try:
            subprocess.run(
                [sys.executable, "-m", "docformatter", "--version"],
                capture_output=True,
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

