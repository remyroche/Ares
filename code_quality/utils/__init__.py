"""Utility functions for code analysis."""

from .file_utils import (
    find_python_files,
    read_file_safely,
    parse_ast_safely
    # extract_function_name_from_issue,  # Removed - unused
    # get_module_from_file_path,  # Removed - unused
    # is_documentation_file  # Removed - unused
)

__all__ = [
    "find_python_files",
    "read_file_safely",
    "parse_ast_safely"
    # "extract_function_name_from_issue",  # Removed - unused
    # "get_module_from_file_path",  # Removed - unused
    # "is_documentation_file"  # Removed - unused
]

