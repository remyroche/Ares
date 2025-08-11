#!/usr/bin/env python3
"""
Script to update logging messages in training step files with warning symbols.

This script automatically adds warning symbols to error and warning messages
throughout the training step files to make issues more visible.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

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


def get_warning_symbol_function(message: str) -> str:
    """
    Determine the appropriate warning symbol function based on the message content.
    
    Args:
        message: The error/warning message
        
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
        # Default to error for error messages, warning for warning messages
        return 'error'


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
        
        # Pattern to match logger.error, logger.warning, logger.exception, logger.critical calls
        # Also match print statements with error/warning indicators
        patterns = [
            # print(error("message")))
            (r'logger\.error\(f?"([^"]*)"', r'logger.error(\1("\1")'),
            # print(warning("message"))) 
            (r'logger\.warning\(f?"([^"]*)"', r'logger.warning(\1("\1")'),
            # print(error("message")))
            (r'logger\.exception\(f?"([^"]*)"', r'logger.exception(\1("\1")'),
            # print(error("message")))
            (r'logger\.critical\(f?"([^"]*)"', r'logger.critical(\1("\1")'),
            # print(warning("message"))) -> print(failed("message"))
            (r'print\(f?"❌ ([^"]*)"', r'print(failed("\1")'),
            # print(warning("message"))) -> print(warning("message"))
            (r'print\(f?"⚠️ ([^"]*)"', r'print(warning("\1")'),
            # print(error("message"))) -> print(error("message"))
            (r'print\(f?"🚨 ([^"]*)"', r'print(error("\1")'),
        ]
        
        for pattern, replacement in patterns:
            # Find all matches
            matches = re.finditer(pattern, content)
            for match in matches:
                message = match.group(1)
                warning_func = get_warning_symbol_function(message)
                
                # Create the replacement
                if 'logger.' in pattern:
                    new_call = f'logger.{match.group(0).split(".")[1].split("(")[0]}({warning_func}("{message}"))'
                else:
                    new_call = f'print({warning_func}("{message}"))'
                
                content = content.replace(match.group(0), new_call)
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
            print(warning(" Could not find logger import in {file_path}")))
            return False
            
    except Exception as e:
        print(warning("Error adding import to {file_path}: {e}")))
        return False


def main():
    """Main function to update all training step files."""
    training_steps_dir = project_root / "src" / "training" / "steps"
    
    if not training_steps_dir.exists():
        print(missing("Training steps directory not found: {training_steps_dir}")))
        return
    
    # Get all Python files in the training steps directory
    python_files = list(training_steps_dir.glob("*.py"))
    
    print(f"🔍 Found {len(python_files)} Python files in training steps directory")
    
    total_changes = 0
    total_files_processed = 0
    
    for file_path in python_files:
        print(f"\n📁 Processing {file_path.name}...")
        
        # Add warning symbols import if needed
        import_added = add_warning_symbols_import(str(file_path))
        
        # Update logging messages
        changes, lines = update_file_logging_messages(str(file_path))
        
        total_changes += changes
        if import_added:
            total_changes += 1
        total_files_processed += 1
    
    print(f"\n✅ Summary:")
    print(f"   Files processed: {total_files_processed}")
    print(f"   Total changes made: {total_changes}")
    print(f"   Average changes per file: {total_changes / total_files_processed:.1f}")


if __name__ == "__main__":
    main() 