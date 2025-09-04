"""
File Utilities for Code Complexity Analysis
"""

import os
import logging
from typing import List, Set
from pathlib import Path

logger = logging.getLogger(__name__)


class FileUtils:
    """Utility class for file operations"""
    
    def __init__(self):
        """Initialize FileUtils"""
        self.python_extensions = {'.py'}
        self.ignore_patterns = {
            '__pycache__',
            '.git',
            '.pytest_cache',
            'venv',
            'env',
            '.venv',
            '.env',
            'node_modules',
            '.tox',
            'build',
            'dist',
            '*.egg-info'
        }
        
    def get_python_files(self, directory_path: str, recursive: bool = True) -> List[str]:
        """Get all Python files in a directory"""
        python_files = []
        
        if not os.path.exists(directory_path):
            logger.error(f"Directory does not exist: {directory_path}")
            return python_files
            
        if os.path.isfile(directory_path):
            if self._is_python_file(directory_path):
                return [directory_path]
            else:
                return []
                
        if recursive:
            for root, dirs, files in os.walk(directory_path):
                # Filter out ignored directories
                dirs[:] = [d for d in dirs if not self._should_ignore_directory(d)]
                
                for file in files:
                    file_path = os.path.join(root, file)
                    if self._is_python_file(file_path):
                        python_files.append(file_path)
        else:
            try:
                for item in os.listdir(directory_path):
                    item_path = os.path.join(directory_path, item)
                    if os.path.isfile(item_path) and self._is_python_file(item_path):
                        python_files.append(item_path)
            except PermissionError:
                logger.error(f"Permission denied accessing directory: {directory_path}")
                
        return sorted(python_files)
        
    def _is_python_file(self, file_path: str) -> bool:
        """Check if a file is a Python file"""
        if not os.path.isfile(file_path):
            return False
            
        # Check extension
        ext = Path(file_path).suffix.lower()
        if ext in self.python_extensions:
            return True
            
        # Check shebang for Python files without extension
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if first_line.startswith('#!') and 'python' in first_line:
                    return True
        except (UnicodeDecodeError, PermissionError):
            pass
            
        return False
        
    def _should_ignore_directory(self, directory_name: str) -> bool:
        """Check if a directory should be ignored"""
        return directory_name in self.ignore_patterns
        
    def get_file_size(self, file_path: str) -> int:
        """Get file size in bytes"""
        try:
            return os.path.getsize(file_path)
        except OSError:
            return 0
            
    def get_file_line_count(self, file_path: str) -> int:
        """Get line count of a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        except (UnicodeDecodeError, PermissionError, OSError):
            return 0
            
    def get_directory_stats(self, directory_path: str) -> dict:
        """Get statistics for a directory"""
        python_files = self.get_python_files(directory_path)
        
        total_lines = 0
        total_size = 0
        
        for file_path in python_files:
            total_lines += self.get_file_line_count(file_path)
            total_size += self.get_file_size(file_path)
            
        return {
            'file_count': len(python_files),
            'total_lines': total_lines,
            'total_size_bytes': total_size,
            'average_lines_per_file': total_lines / len(python_files) if python_files else 0,
            'average_size_per_file': total_size / len(python_files) if python_files else 0
        }
        
    def find_large_files(self, directory_path: str, min_size_mb: float = 1.0) -> List[dict]:
        """Find files larger than specified size"""
        large_files = []
        min_size_bytes = min_size_mb * 1024 * 1024
        
        python_files = self.get_python_files(directory_path)
        
        for file_path in python_files:
            size = self.get_file_size(file_path)
            if size > min_size_bytes:
                large_files.append({
                    'file_path': file_path,
                    'size_bytes': size,
                    'size_mb': size / (1024 * 1024),
                    'line_count': self.get_file_line_count(file_path)
                })
                
        return sorted(large_files, key=lambda x: x['size_bytes'], reverse=True)
        
    def find_long_files(self, directory_path: str, min_lines: int = 1000) -> List[dict]:
        """Find files with more than specified lines"""
        long_files = []
        
        python_files = self.get_python_files(directory_path)
        
        for file_path in python_files:
            line_count = self.get_file_line_count(file_path)
            if line_count > min_lines:
                long_files.append({
                    'file_path': file_path,
                    'line_count': line_count,
                    'size_bytes': self.get_file_size(file_path),
                    'size_mb': self.get_file_size(file_path) / (1024 * 1024)
                })
                
        return sorted(long_files, key=lambda x: x['line_count'], reverse=True)