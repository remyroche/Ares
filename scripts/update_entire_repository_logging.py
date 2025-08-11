#!/usr/bin/env python3
"""
Comprehensive script to update logging messages throughout the entire repository with warning symbols.

This script automatically adds warning symbols to error and warning messages
throughout the entire Ares trading bot codebase to make issues more visible.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Set

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.warning_symbols import (
    error,
    warning,
    critical,
    problem,
    failed,
    invalid,
    missing,
    timeout,
    connection_error,
    validation_error,
    initialization_error,
    execution_error,
)


def get_warning_symbol_function(message: str, log_level: str = "error") -> str:
    """
    Determine the appropriate warning symbol function based on the message content.
    
    Args:
        message: The error/warning message
        log_level: The logging level (error, warning, exception, critical)
        
    Returns:
        The appropriate warning symbol function name
    """
    message_lower = message.lower()
    
    # Error patterns
    if any(word in message_lower for word in ['failed', 'failure', 'fail']):
        return 'failed'
    elif any(word in message_lower for word in ['invalid', 'invalid configuration']):
        return 'invalid'
    elif any(word in message_lower for word in ['missing', 'not found', 'file not found']):
        return 'missing'
    elif any(word in message_lower for word in ['timeout', 'timed out']):
        return 'timeout'
    elif any(word in message_lower for word in ['connection', 'network']):
        return 'connection_error'
    elif any(word in message_lower for word in ['validation', 'validate']):
        return 'validation_error'
    elif any(word in message_lower for word in ['initialization', 'init', 'initialize']):
        return 'initialization_error'
    elif any(word in message_lower for word in ['execution', 'execute', 'runtime']):
        return 'execution_error'
    elif any(word in message_lower for word in ['critical', 'fatal']):
        return 'critical'
    elif any(word in message_lower for word in ['problem', 'issue']):
        return 'problem'
    else:
        # Default based on log level
        if log_level in ['error', 'exception', 'critical']:
            return 'error'
        else:
            return 'warning'


def should_skip_file(file_path: str) -> bool:
    """
    Check if a file should be skipped from processing.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file should be skipped, False otherwise
    """
    # Skip certain directories and files
    skip_patterns = [
        '__pycache__',
        '.git',
        '.DS_Store',
        'node_modules',
        'venv',
        'env',
        '.pytest_cache',
        'build',
        'dist',
        '*.pyc',
        '*.log',
        '*.tmp',
        '*.bak',
        '*.swp',
        '*.swo',
        '*.orig',
        '*.rej',
        '*.patch',
        '*.diff',
        '*.md',
        '*.txt',
        '*.json',
        '*.yaml',
        '*.yml',
        '*.toml',
        '*.cfg',
        '*.ini',
        '*.conf',
        '*.sh',
        '*.bat',
        '*.ps1',
        '*.sql',
        '*.html',
        '*.css',
        '*.js',
        '*.ts',
        '*.jsx',
        '*.tsx',
        '*.vue',
        '*.svelte',
        '*.rs',
        '*.go',
        '*.java',
        '*.cpp',
        '*.c',
        '*.h',
        '*.hpp',
        '*.cs',
        '*.php',
        '*.rb',
        '*.pl',
        '*.pyx',
        '*.pxd',
        '*.pxi',
        '*.pyd',
        '*.so',
        '*.dll',
        '*.dylib',
        '*.exe',
        '*.bin',
        '*.dat',
        '*.csv',
        '*.tsv',
        '*.xlsx',
        '*.xls',
        '*.db',
        '*.sqlite',
        '*.sqlite3',
        '*.rdb',
        '*.aof',
        '*.pid',
        '*.lock',
        '*.tmp',
        '*.cache',
        '*.pyc',
        '*.pyo',
        '*.pyd',
        '*.so',
        '*.dll',
        '*.dylib',
        '*.exe',
        '*.bin',
        '*.dat',
        '*.csv',
        '*.tsv',
        '*.xlsx',
        '*.xls',
        '*.db',
        '*.sqlite',
        '*.sqlite3',
        '*.rdb',
        '*.aof',
        '*.pid',
        '*.lock',
        '*.tmp',
        '*.cache',
    ]
    
    file_path_lower = file_path.lower()
    
    # Check if file matches any skip pattern
    for pattern in skip_patterns:
        if pattern.startswith('*.'):
            # File extension pattern
            if file_path_lower.endswith(pattern[1:]):
                return True
        elif pattern in file_path_lower:
            return True
    
    return False


def update_file_logging_messages(file_path: str) -> Tuple[int, int]:
    """
    Update logging messages in a file with warning symbols.
    
    Args:
        file_path: Path to the file to update
        
    Returns:
        Tuple of (number of changes made, number of lines processed)
    """
    changes_made = 0
    lines_processed = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Patterns to match various logging and print statements
        patterns = [
            # print(error("message")))
            (r'logger\.error\(f?"([^"]*)"', 'error'),
            # print(warning("message"))) 
            (r'logger\.warning\(f?"([^"]*)"', 'warning'),
            # print(error("message")))
            (r'logger\.exception\(f?"([^"]*)"', 'exception'),
            # print(error("message")))
            (r'logger\.critical\(f?"([^"]*)"', 'critical'),
            # print(warning("message"))) -> print(failed("message"))
            (r'print\(f?"❌ ([^"]*)"', 'failed'),
            # print(warning("message"))) -> print(warning("message"))
            (r'print\(f?"⚠️ ([^"]*)"', 'warning'),
            # print(error("message"))) -> print(error("message"))
            (r'print\(f?"🚨 ([^"]*)"', 'error'),
            # print(error("message"))) -> print(critical("message"))
            (r'print\(f?"💥 ([^"]*)"', 'critical'),
            # print(warning("message"))) -> print(problem("message"))
            (r'print\(f?"🔴 ([^"]*)"', 'problem'),
            # print(warning("message"))) -> print(warning("message"))
            (r'print\(f?"❗ ([^"]*)"', 'warning'),
            # print(warning("message"))) -> print(warning("message"))
            (r'print\(f?"‼ ([^"]*)"', 'warning'),
            # print(error("message"))) -> print(error("message"))
            (r'print\(f?"⭕ ([^"]*)"', 'error'),
            # print(error("message"))) -> print(error("message"))
            (r'print\(f?"🛑 ([^"]*)"', 'error'),
            # print(error("message"))) -> print(critical("message"))
            (r'print\(f?"💀 ([^"]*)"', 'critical'),
            # print(error("message"))) -> print(critical("message"))
            (r'print\(f?"💣 ([^"]*)"', 'critical'),
            # print(warning("message"))) -> print(problem("message"))
            (r'print\(f?"● ([^"]*)"', 'problem'),
            # print(warning("message"))) -> print(problem("message"))
            (r'print\(f?"🔴 ([^"]*)"', 'problem'),
        ]
        
        for pattern, default_func in patterns:
            # Find all matches
            matches = list(re.finditer(pattern, content))
            for match in reversed(matches):  # Process in reverse to avoid index issues
                message = match.group(1)
                warning_func = get_warning_symbol_function(message, default_func)
                
                # Create the replacement
                if 'logger.' in pattern:
                    log_method = pattern.split('.')[1].split('(')[0]
                    new_call = f'logger.{log_method}({warning_func}("{message}"))'
                else:
                    new_call = f'print({warning_func}("{message}"))'
                
                # Replace the match
                start, end = match.span()
                content = content[:start] + new_call + content[end:]
                changes_made += 1
        
        # Only write if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Updated {file_path} with {changes_made} changes")
        else:
            print(f"ℹ️  No changes needed for {file_path}")
            
        return changes_made, len(content.split('\n'))
        
    except Exception as e:
        print(warning("Error processing {file_path}: {e}")))
        return 0, 0


def add_warning_symbols_import(file_path: str) -> bool:
    """
    Add warning symbols import to a file if it doesn't already have it.
    
    Args:
        file_path: Path to the file to update
        
    Returns:
        True if import was added, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if warning symbols are already imported
        if 'from src.utils.warning_symbols import' in content:
            return False
        
        # Find the logger import line
        logger_import_pattern = r'from src\.utils\.logger import.*'
        match = re.search(logger_import_pattern, content)
        
        if match:
            # Add warning symbols import after logger import
            warning_import = '''from src.utils.warning_symbols import (
    error,
    warning,
    critical,
    problem,
    failed,
    invalid,
    missing,
    timeout,
    connection_error,
    validation_error,
    initialization_error,
    execution_error,
)'''
            
            # Insert after the logger import
            new_content = content.replace(match.group(0), match.group(0) + '\n' + warning_import)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print(f"✅ Added warning symbols import to {file_path}")
            return True
        else:
            # Try to find any import line to add after
            import_pattern = r'^import .*$|^from .* import .*$'
            lines = content.split('\n')
            
            for i, line in enumerate(lines):
                if re.match(import_pattern, line.strip()):
                    # Add warning symbols import after this import
                    warning_import = '''from src.utils.warning_symbols import (
    error,
    warning,
    critical,
    problem,
    failed,
    invalid,
    missing,
    timeout,
    connection_error,
    validation_error,
    initialization_error,
    execution_error,
)'''
                    
                    lines.insert(i + 1, warning_import)
                    new_content = '\n'.join(lines)
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    
                    print(f"✅ Added warning symbols import to {file_path}")
                    return True
            
            print(warning(" Could not find suitable import location in {file_path}")))
            return False
            
    except Exception as e:
        print(warning("Error adding import to {file_path}: {e}")))
        return False


def find_python_files(directory: Path) -> List[Path]:
    """
    Recursively find all Python files in a directory.
    
    Args:
        directory: Directory to search
        
    Returns:
        List of Python file paths
    """
    python_files = []
    
    try:
        for item in directory.rglob("*.py"):
            if not should_skip_file(str(item)):
                python_files.append(item)
    except Exception as e:
        print(warning("Error searching directory {directory}: {e}")))
    
    return python_files


def main():
    """Main function to update all Python files in the repository."""
    print("🚀 Starting comprehensive repository logging update...")
    print(f"📁 Project root: {project_root}")
    
    # Find all Python files in the repository
    python_files = find_python_files(project_root)
    
    print(f"🔍 Found {len(python_files)} Python files to process")
    
    total_changes = 0
    total_files_processed = 0
    files_with_imports_added = 0
    
    for file_path in python_files:
        print(f"\n📁 Processing {file_path.relative_to(project_root)}...")
        
        # Add warning symbols import if needed
        import_added = add_warning_symbols_import(str(file_path))
        if import_added:
            files_with_imports_added += 1
        
        # Update logging messages
        changes, lines = update_file_logging_messages(str(file_path))
        
        total_changes += changes
        if import_added:
            total_changes += 1
        total_files_processed += 1
    
    print(f"\n✅ Summary:")
    print(f"   Files processed: {total_files_processed}")
    print(f"   Files with imports added: {files_with_imports_added}")
    print(f"   Total changes made: {total_changes}")
    if total_files_processed > 0:
        print(f"   Average changes per file: {total_changes / total_files_processed:.1f}")
    
    print(f"\n🎉 Repository logging update completed successfully!")


if __name__ == "__main__":
    main() 