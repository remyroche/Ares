#!/usr/bin/env python3
"""Fix common syntax errors in Python files."""

import json
import re
import shutil
from pathlib import Path


def fix_duplicate_imports(content, file_path):
    """Fix duplicate import aliases like 'handles_errors as handles_errors_src_core_decorators as core_handles_errors'."""
    # Pattern to match the problematic imports
    pattern = r'from\s+src\.core\.decorators\s+import\s+handles_errors\s+as\s+handles_errors_src_core_decorators\s+as\s+core_handles_errors\s+as\s+core_handles_errors'
    
    # Replace with correct import
    fixed = re.sub(pattern, 'from src.core.decorators import handles_errors', content)
    
    # Also fix similar patterns
    pattern2 = r'(\w+)\s+as\s+\1_src_core_decorators\s+as\s+core_\1\s+as\s+core_\1'
    fixed = re.sub(pattern2, r'\1', fixed)
    
    return fixed


def fix_missing_try_block(content, line_num):
    """Add missing content after try: statement."""
    lines = content.split('\n')
    
    # Find the try statement
    for i in range(line_num - 1, min(line_num + 5, len(lines))):
        if i < len(lines) and lines[i].strip().startswith('try:'):
            # Check if next line is empty or not properly indented
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                indent = len(lines[i]) - len(lines[i].lstrip())
                
                if not next_line.strip() or not next_line.startswith(' ' * (indent + 4)):
                    # Add a pass statement
                    lines.insert(i + 1, ' ' * (indent + 4) + 'pass  # TODO: Add try block content')
                    return '\n'.join(lines)
    
    return content


def fix_missing_except_block(content, error_line):
    """Add missing except or finally block."""
    lines = content.split('\n')
    
    # Search backwards from error line to find the try block
    try_line = None
    try_indent = 0
    
    for i in range(error_line - 1, max(0, error_line - 50), -1):
        if lines[i].strip().startswith('try:'):
            try_line = i
            try_indent = len(lines[i]) - len(lines[i].lstrip())
            break
    
    if try_line is not None:
        # Find where to insert the except block
        insert_line = try_line + 1
        
        # Skip over the try block content
        while insert_line < len(lines) and (
            not lines[insert_line].strip() or 
            lines[insert_line].startswith(' ' * (try_indent + 4))
        ):
            insert_line += 1
        
        # Add except block
        except_block = [
            ' ' * try_indent + 'except Exception as e:',
            ' ' * (try_indent + 4) + 'pass  # TODO: Handle exception'
        ]
        
        for j, line in enumerate(except_block):
            lines.insert(insert_line + j, line)
        
        return '\n'.join(lines)
    
    return content


def fix_unexpected_indent(content, error_line):
    """Fix unexpected indent by aligning with previous lines."""
    lines = content.split('\n')
    
    if error_line - 1 < len(lines):
        problem_line = lines[error_line - 1]
        problem_indent = len(problem_line) - len(problem_line.lstrip())
        
        # Look at previous non-empty lines to determine correct indent
        for i in range(error_line - 2, max(0, error_line - 10), -1):
            if lines[i].strip():
                prev_indent = len(lines[i]) - len(lines[i].lstrip())
                
                # If previous line ends with : it should indent by 4
                if lines[i].rstrip().endswith(':'):
                    correct_indent = prev_indent + 4
                else:
                    correct_indent = prev_indent
                
                # Fix the indent
                lines[error_line - 1] = ' ' * correct_indent + problem_line.lstrip()
                return '\n'.join(lines)
    
    return content


def fix_file(file_path, error_info):
    """Fix syntax errors in a file."""
    try:
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create backup
        backup_path = file_path + '.syntax_backup'
        shutil.copy2(file_path, backup_path)
        
        original_content = content
        
        # Apply fixes based on error type
        error_msg = error_info['msg']
        error_line = error_info.get('line', 0)
        
        if 'handles_errors_src_core_decorators' in content:
            content = fix_duplicate_imports(content, file_path)
        
        elif 'expected an indented block after \'try\' statement' in error_msg:
            content = fix_missing_try_block(content, error_line)
        
        elif 'expected \'except\' or \'finally\' block' in error_msg:
            content = fix_missing_except_block(content, error_line)
        
        elif 'unexpected indent' in error_msg:
            content = fix_unexpected_indent(content, error_line)
        
        # Write fixed content if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, "Fixed"
        else:
            return False, "No automatic fix available"
            
    except Exception as e:
        return False, f"Error: {str(e)}"


def main():
    """Main function to fix syntax errors."""
    # Load syntax errors
    with open('/workspace/actual_syntax_errors.json', 'r') as f:
        data = json.load(f)
    
    errors = data['errors']
    
    # Fix high-priority files first
    priority_files = [
        'ml_target_validator.py',
        'ml_tactics_manager.py',
        'position_sizer.py',
        'enhanced_trading_launcher.py',
        'model_trainer.py',
        'enhanced_training_manager.py'
    ]
    
    # Sort errors by priority
    sorted_errors = []
    for pf in priority_files:
        for error in errors:
            if pf in error['file']:
                sorted_errors.append(error)
    
    # Add remaining errors
    for error in errors:
        if error not in sorted_errors:
            sorted_errors.append(error)
    
    # Fix files
    fixed_count = 0
    failed_count = 0
    results = []
    
    print("Fixing syntax errors...\n")
    
    for i, error in enumerate(sorted_errors[:20]):  # Fix first 20 files
        file_path = error['file']
        print(f"{i+1}. Fixing {file_path}...")
        
        success, message = fix_file(file_path, error)
        
        if success:
            fixed_count += 1
            print(f"   ✓ {message}")
        else:
            failed_count += 1
            print(f"   ✗ {message}")
        
        results.append({
            'file': file_path,
            'success': success,
            'message': message,
            'error': error
        })
    
    print(f"\n{'='*60}")
    print(f"Fixed: {fixed_count} files")
    print(f"Failed: {failed_count} files")
    print(f"Remaining: {len(errors) - 20} files")
    
    # Save results
    with open('/workspace/syntax_fix_results.json', 'w') as f:
        json.dump({
            'fixed': fixed_count,
            'failed': failed_count,
            'results': results
        }, f, indent=2)
    
    print(f"\nResults saved to: /workspace/syntax_fix_results.json")


if __name__ == "__main__":
    main()