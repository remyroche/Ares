#!/usr/bin/env python3
from src.utils.tprint import tprint

"""Gitignore pattern parser and matcher for code analysis pipelines."""

import fnmatch
import re
from pathlib import Path
from typing import List, Set, Optional


class GitignoreParser:
    """Parser for .gitignore patterns with support for complex matching rules."""
    
    def __init__(self, project_root: Path):
        """
        Initialize the gitignore parser.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.patterns: List[tuple] = []  # (pattern, is_negation, is_directory)
        self._load_gitignore_patterns()
    
    def _load_gitignore_patterns(self) -> None:
        """Load patterns from .gitignore file."""
        gitignore_path = self.project_root / '.gitignore'
        
        if not gitignore_path.exists():
            return
        
        try:
            with open(gitignore_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    # Handle negation patterns (starting with !)
                    is_negation = line.startswith('!')
                    if is_negation:
                        pattern = line[1:]
                    else:
                        pattern = line
                    
                    # Handle directory patterns (ending with /)
                    is_directory = pattern.endswith('/')
                    if is_directory:
                        pattern = pattern[:-1]
                    
                    # Skip if pattern becomes empty after processing
                    if not pattern:
                        continue
                    
                    self.patterns.append((pattern, is_negation, is_directory))
                    
        except Exception as e:
            tprint(f"Warning: Could not read .gitignore file: {e}")
    
    def should_ignore(self, file_path: Path) -> bool:
        """
        Check if a file or directory should be ignored based on .gitignore patterns.
        
        Args:
            file_path: Path to check (can be file or directory)
            
        Returns:
            True if the path should be ignored, False otherwise
        """
        # Convert to relative path from project root
        try:
            relative_path = file_path.relative_to(self.project_root)
        except ValueError:
            # File is outside project root, don't ignore
            return False
        
        relative_path_str = str(relative_path).replace('\\', '/')
        path_parts = relative_path.parts
        
        # Track negation matches
        negation_matches: Set[str] = set()
        
        for pattern, is_negation, is_directory in self.patterns:
            # Convert pattern to use forward slashes
            pattern = pattern.replace('\\', '/')
            
            # Handle directory patterns
            if is_directory:
                # For directory patterns, check if any parent directory matches
                for i in range(len(path_parts)):
                    parent_path = '/'.join(path_parts[:i+1])
                    if self._match_pattern(pattern, parent_path):
                        if is_negation:
                            negation_matches.add(parent_path)
                        else:
                            # If there's a negation match for this path, don't ignore
                            if parent_path not in negation_matches:
                                return True
            else:
                # For file patterns, check the full path and filename
                if self._match_pattern(pattern, relative_path_str):
                    if is_negation:
                        negation_matches.add(relative_path_str)
                    else:
                        # If there's a negation match for this path, don't ignore
                        if relative_path_str not in negation_matches:
                            return True
                
                # Also check just the filename
                filename = path_parts[-1] if path_parts else ""
                if self._match_pattern(pattern, filename):
                    if is_negation:
                        negation_matches.add(filename)
                    else:
                        # If there's a negation match for this filename, don't ignore
                        if filename not in negation_matches:
                            return True
        
        return False
    
    def _match_pattern(self, pattern: str, path: str) -> bool:
        """
        Match a gitignore pattern against a path.
        
        Args:
            pattern: Gitignore pattern
            path: Path to match against
            
        Returns:
            True if pattern matches, False otherwise
        """
        # Handle special cases
        if pattern == '*':
            return True
        
        # Convert gitignore pattern to fnmatch pattern
        # Handle ** (matches any number of directories)
        if '**' in pattern:
            # Replace ** with * for fnmatch
            pattern = pattern.replace('**', '*')
        
        # Handle leading slash (matches from root)
        if pattern.startswith('/'):
            pattern = pattern[1:]
            # Only match if path starts with the pattern
            return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(path, pattern + '/*')
        
        # Handle patterns that don't start with /
        # These can match anywhere in the path
        path_parts = path.split('/')
        pattern_parts = pattern.split('/')
        
        # If pattern has multiple parts, use fnmatch directly
        if len(pattern_parts) > 1:
            return fnmatch.fnmatch(path, pattern)
        
        # Single part pattern - check if it matches any part of the path
        for part in path_parts:
            if fnmatch.fnmatch(part, pattern):
                return True
        
        return False
    
    def get_ignored_directories(self) -> Set[str]:
        """
        Get a set of directory patterns that should be ignored.
        
        Returns:
            Set of directory names/patterns to ignore
        """
        ignored_dirs = set()
        
        for pattern, is_negation, is_directory in self.patterns:
            if is_negation:
                continue
            
            if is_directory:
                # Extract directory name from pattern
                if '/' in pattern:
                    dir_name = pattern.split('/')[-1]
                else:
                    dir_name = pattern
                ignored_dirs.add(dir_name)
            else:
                # Check if pattern looks like a directory pattern
                if pattern.endswith('*') and not pattern.endswith('.*'):
                    # Likely a directory pattern like "build*"
                    dir_name = pattern.rstrip('*')
                    if dir_name:
                        ignored_dirs.add(dir_name)
        
        return ignored_dirs


def should_ignore_file(file_path: Path, project_root: Optional[Path] = None) -> bool:
    """
    Convenience function to check if a file should be ignored.
    
    Args:
        file_path: Path to the file to check
        project_root: Project root directory (defaults to finding .gitignore in parent dirs)
        
    Returns:
        True if file should be ignored, False otherwise
    """
    if project_root is None:
        # Find project root by looking for .gitignore
        current = file_path.parent
        while current != current.parent:
            if (current / '.gitignore').exists():
                project_root = current
                break
            current = current.parent
        
        if project_root is None:
            return False
    
    parser = GitignoreParser(project_root)
    return parser.should_ignore(file_path)


def filter_ignored_files(file_paths: List[Path], project_root: Optional[Path] = None) -> List[Path]:
    """
    Filter out files that should be ignored according to .gitignore.
    
    Args:
        file_paths: List of file paths to filter
        project_root: Project root directory
        
    Returns:
        List of file paths that are not ignored
    """
    if not file_paths:
        return []
    
    if project_root is None:
        # Use the parent directory of the first file as project root
        project_root = file_paths[0].parent
        while project_root != project_root.parent:
            if (project_root / '.gitignore').exists():
                break
            project_root = project_root.parent
    
    parser = GitignoreParser(project_root)
    return [path for path in file_paths if not parser.should_ignore(path)]