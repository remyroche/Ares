"""
Pyre type checker plugin for code quality tools.
"""

import subprocess
import sys
import json
from typing import Dict, Any, List

from code_quality.core.plugins import BaseCodeAnalyzer


class PyreAnalyzer(BaseCodeAnalyzer):
    """Pyre static type checker plugin."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "pyre"
        self.description = "Static type checker for Python (Facebook Pyre)"
        self.version = "1.0.0"

    def get_name(self) -> str:
        return self.name

    def get_description(self) -> str:
        return self.description

    def get_version(self) -> str:
        return self.version

    def can_analyze(self, file_path: str) -> bool:
        # Pyre typically runs project-wide; allow per-file but it will still run project context
        return file_path.endswith('.py')

    def analyze(self, file_path: str) -> Dict[str, Any]:
        """Analyze types using Pyre. Expects a Pyre-initialized project (.pyre configuration)."""
        try:
            cmd: List[str] = [
                sys.executable, "-m", "pyre_check", "check", "--output", "json"
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode in (0, 1, 2):
                issues = []
                if result.stdout.strip():
                    try:
                        # pyre outputs JSON per line; attempt to parse array or lines
                        try:
                            issues = json.loads(result.stdout)
                        except json.JSONDecodeError:
                            issues = [json.loads(line) for line in result.stdout.splitlines() if line.strip().startswith('{')]
                    except Exception:
                        issues = []
                return {
                    'success': True,
                    'tool': 'pyre',
                    'file': file_path,
                    'issues': issues,
                    'total_issues': len(issues),
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
            else:
                return {
                    'success': False,
                    'tool': 'pyre',
                    'file': file_path,
                    'message': f'Pyre failed with return code {result.returncode}',
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'error': True
                }

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'tool': 'pyre',
                'file': file_path,
                'message': 'Pyre timed out',
                'error': True,
                'timeout': True
            }
        except Exception as e:
            return {
                'success': False,
                'tool': 'pyre',
                'file': file_path,
                'message': f'Pyre failed with exception: {str(e)}',
                'error': True,
                'exception': str(e)
            }

    def get_supported_extensions(self) -> List[str]:
        return ['.py']

    def check_installed(self) -> bool:
        try:
            subprocess.run(
                [sys.executable, "-m", "pyre_check", "--version"],
                capture_output=True,
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

