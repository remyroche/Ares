"""
unify plugin for normalizing string quotes and formatting nuances.
"""

import subprocess
import sys
from typing import Dict, Any, List

from code_quality.core.plugins import BaseCodeFixer


class UnifyFixer(BaseCodeFixer):
    """unify plugin to normalize quotes and minor formatting."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "unify"
        self.description = "Normalizes string quotes and whitespace using unify"
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
        """Normalize code using unify."""
        try:
            # Prefer double quotes to reduce escaping of apostrophes
            cmd: List[str] = [
                sys.executable, "-m", "unify",
                "-i",
                "--quote", '"',
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
                    'tool': 'unify',
                    'file': file_path,
                    'message': 'Successfully normalized with unify',
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
            else:
                return {
                    'success': False,
                    'tool': 'unify',
                    'file': file_path,
                    'message': f'unify failed with return code {result.returncode}',
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'error': True
                }

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'tool': 'unify',
                'file': file_path,
                'message': 'unify timed out after 60 seconds',
                'error': True,
                'timeout': True
            }
        except Exception as e:
            return {
                'success': False,
                'tool': 'unify',
                'file': file_path,
                'message': f'unify failed with exception: {str(e)}',
                'error': True,
                'exception': str(e)
            }

    def get_supported_extensions(self) -> List[str]:
        return ['.py']

    def check_installed(self) -> bool:
        try:
            subprocess.run(
                [sys.executable, "-m", "unify", "--version"],
                capture_output=True,
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

