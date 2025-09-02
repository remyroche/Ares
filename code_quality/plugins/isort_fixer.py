"""
isort import sorter plugin for code quality tools.
"""

import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List
from ..core.plugins import BaseCodeFixer


class IsortFixer(BaseCodeFixer):
    """isort import sorter plugin."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "isort"
        self.description = "Python import sorter and formatter"
        self.version = "1.0.0"
    
    def get_name(self) -> str:
        return self.name
    
    def get_description(self) -> str:
        return self.description
    
    def get_version(self) -> str:
        return self.version
    
    def can_fix(self, file_path: str) -> bool:
        """Check if isort can fix this file."""
        return file_path.endswith('.py')
    
    def fix(self, file_path: str) -> Dict[str, Any]:
        """Fix import organization using isort."""
        try:
            # Build isort command
            cmd = [
                sys.executable, "-m", "isort",
                "--quiet",  # Suppress output
                "--profile", "black",  # Use Black-compatible profile
                "--line-length", str(self.get_config('max_line_length', 88))
            ]
            
            # Add additional options if configured
            if self.get_config('aggressive', False):
                cmd.extend(["--diff", "--check-only"])
            
            cmd.append(file_path)
            
            # Run isort
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'tool': 'isort',
                    'file': file_path,
                    'message': 'Successfully sorted imports with isort',
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
            else:
                return {
                    'success': False,
                    'tool': 'isort',
                    'file': file_path,
                    'message': f'isort failed with return code {result.returncode}',
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'error': True
                }
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'tool': 'isort',
                'file': file_path,
                'message': 'isort timed out after 60 seconds',
                'error': True,
                'timeout': True
            }
        except Exception as e:
            return {
                'success': False,
                'tool': 'isort',
                'file': file_path,
                'message': f'isort failed with exception: {str(e)}',
                'error': True,
                'exception': str(e)
            }
    
    def get_supported_extensions(self) -> List[str]:
        return ['.py']
    
    def check_installed(self) -> bool:
        """Check if isort is installed."""
        try:
            subprocess.run(
                [sys.executable, "-m", "isort", "--version"],
                capture_output=True,
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False