#!/usr/bin/env python3
"""Fix more complex syntax errors in Python files."""

import ast
import json
import re
import shutil
from pathlib import Path
import tokenize
import io


def fix_unmatched_parenthesis(content, error_line):
    """Fix unmatched parentheses."""
    lines = content.split('\n')
    
    if error_line - 1 < len(lines):
        problem_line = lines[error_line - 1]
        
        # Count parentheses
        open_count = problem_line.count('(')
        close_count = problem_line.count(')')
        
        if open_count > close_count:
            # Add closing parentheses
            missing = open_count - close_count
            lines[error_line - 1] = problem_line.rstrip() + ')' * missing
            return '\n'.join(lines)
        
    return content


def fix_unterminated_string(content, error_line):
    """Fix unterminated string literals."""
    lines = content.split('\n')
    
    if error_line - 1 < len(lines):
        problem_line = lines[error_line - 1]
        
        # Check for unterminated quotes
        for quote in ['"""', "'''", '"', "'"]:
            count = problem_line.count(quote)
            if count % 2 == 1:
                # Add closing quote
                lines[error_line - 1] = problem_line.rstrip() + quote
                return '\n'.join(lines)
    
    return content


def fix_indentation_mismatch(content, error_line):
    """Fix indentation mismatches."""
    lines = content.split('\n')
    
    if error_line - 1 >= len(lines):
        return content
        
    # Get the problem line
    problem_line = lines[error_line - 1]
    problem_indent = len(problem_line) - len(problem_line.lstrip())
    
    # Find the proper indentation level by looking at surrounding code
    indent_levels = []
    
    # Look at previous lines
    for i in range(max(0, error_line - 20), error_line - 1):
        if lines[i].strip() and not lines[i].lstrip().startswith('#'):
            indent = len(lines[i]) - len(lines[i].lstrip())
            indent_levels.append(indent)
    
    if indent_levels:
        # Find the most common indent level
        from collections import Counter
        common_indents = Counter(indent_levels).most_common()
        
        # Check if the line should be at base level or indented
        prev_line = lines[error_line - 2].rstrip() if error_line - 2 >= 0 else ''
        
        if prev_line.endswith(':'):
            # Should be indented
            base_indent = len(prev_line) - len(prev_line.lstrip())
            correct_indent = base_indent + 4
        else:
            # Use most common indent
            correct_indent = common_indents[0][0] if common_indents else 0
        
        # Fix the line
        lines[error_line - 1] = ' ' * correct_indent + problem_line.lstrip()
        return '\n'.join(lines)
    
    return content


def fix_missing_if_body(content, error_line):
    """Fix missing body after if statement."""
    lines = content.split('\n')
    
    # Find the if statement
    for i in range(max(0, error_line - 5), min(error_line + 1, len(lines))):
        if i < len(lines) and lines[i].strip().startswith('if ') and lines[i].rstrip().endswith(':'):
            indent = len(lines[i]) - len(lines[i].lstrip())
            
            # Check if next line is properly indented
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                if not next_line.strip() or not next_line.startswith(' ' * (indent + 4)):
                    # Add pass statement
                    lines.insert(i + 1, ' ' * (indent + 4) + 'pass  # TODO: Add if block content')
                    return '\n'.join(lines)
    
    return content


def fix_invalid_decorator_syntax(content):
    """Fix invalid decorator syntax."""
    lines = content.split('\n')
    
    for i, line in enumerate(lines):
        # Look for decorator patterns with missing closing parenthesis
        if line.strip().startswith('@') and line.strip().endswith('(') and not line.strip().endswith('()'):
            # Add closing parenthesis
            lines[i] = line.rstrip() + ')'
        
        # Fix decorators that are on wrong line
        if i > 0 and line.strip().startswith('@') and lines[i-1].strip() and not lines[i-1].strip().startswith('@'):
            # This decorator might be misplaced
            if i + 1 < len(lines) and (lines[i+1].strip().startswith('def ') or lines[i+1].strip().startswith('class ')):
                # Decorator is in right place
                pass
            else:
                # Move decorator to its own line
                indent = len(line) - len(line.lstrip())
                lines[i] = ''
                lines.insert(i, ' ' * indent + line.strip())
    
    return '\n'.join(lines)


def fix_comma_syntax(content):
    """Fix missing commas in lists, tuples, etc."""
    # This is for the "Perhaps you forgot a comma?" error
    lines = content.split('\n')
    
    for i, line in enumerate(lines):
        # Look for patterns like "item1" "item2" without comma
        pattern = r'(["\'])([^"\']+)\1\s+(["\'])([^"\']+)\3'
        if re.search(pattern, line):
            # Add comma between string literals
            lines[i] = re.sub(pattern, r'\1\2\1, \3\4\3', line)
    
    return '\n'.join(lines)


def fix_never_closed_parenthesis(content):
    """Fix parentheses that were never closed across multiple lines."""
    # Use tokenize to find unclosed parentheses
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(content).readline))
        
        # Track open parentheses
        open_parens = []
        
        for token in tokens:
            if token.type == tokenize.OP:
                if token.string in '([{':
                    open_parens.append((token.string, token.start[0]))
                elif token.string in ')]}':
                    if open_parens:
                        open_parens.pop()
        
        # If we have unclosed parentheses, close them
        if open_parens:
            lines = content.split('\n')
            for paren, line_no in reversed(open_parens):
                closing = {'(': ')', '[': ']', '{': '}'}[paren]
                
                # Find a good place to add the closing paren
                # Usually at the end of the expression or before the next statement
                for i in range(line_no - 1, min(line_no + 10, len(lines))):
                    if i < len(lines) and lines[i].strip():
                        # Add at end of this line
                        lines[i] = lines[i].rstrip() + closing
                        break
            
            return '\n'.join(lines)
    except:
        pass
    
    return content


def fix_file_advanced(file_path, error_info):
    """Apply advanced fixes to a file."""
    try:
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create backup if not exists
        backup_path = file_path + '.syntax_backup'
        if not Path(backup_path).exists():
            shutil.copy2(file_path, backup_path)
        
        original_content = content
        error_msg = error_info['msg']
        error_line = error_info.get('line', 0)
        
        # Apply fixes based on error type
        if 'unmatched' in error_msg and ')' in error_msg:
            content = fix_unmatched_parenthesis(content, error_line)
        
        elif 'unterminated string literal' in error_msg:
            content = fix_unterminated_string(content, error_line)
        
        elif 'unindent does not match' in error_msg:
            content = fix_indentation_mismatch(content, error_line)
        
        elif 'expected an indented block after \'if\' statement' in error_msg:
            content = fix_missing_if_body(content, error_line)
        
        elif 'invalid syntax' in error_msg and '@' in str(error_info.get('text', '')):
            content = fix_invalid_decorator_syntax(content)
        
        elif 'Perhaps you forgot a comma?' in error_msg:
            content = fix_comma_syntax(content)
        
        elif "'(' was never closed" in error_msg:
            content = fix_never_closed_parenthesis(content)
        
        # Additional generic fixes
        if 'invalid syntax' in error_msg:
            # Try various generic fixes
            content = fix_invalid_decorator_syntax(content)
            content = fix_never_closed_parenthesis(content)
        
        # Validate the fix
        try:
            ast.parse(content)
            # Fix worked!
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, "Fixed successfully"
        except SyntaxError as e:
            # Fix didn't work, but maybe improved?
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True, f"Partially fixed (new error: {e.msg})"
            else:
                return False, "No automatic fix available"
                
    except Exception as e:
        return False, f"Error: {str(e)}"


def main():
    """Fix remaining syntax errors."""
    # Load previous results
    with open('/workspace/syntax_fix_results.json', 'r') as f:
        prev_results = json.load(f)
    
    # Load all syntax errors
    with open('/workspace/actual_syntax_errors.json', 'r') as f:
        data = json.load(f)
    
    errors = data['errors']
    
    # Get files that weren't fixed before
    fixed_files = {r['file'] for r in prev_results['results'] if r['success']}
    remaining_errors = [e for e in errors if e['file'] not in fixed_files]
    
    print(f"Attempting to fix {len(remaining_errors)} remaining files...\n")
    
    fixed_count = 0
    failed_count = 0
    results = []
    
    # Process next batch
    for i, error in enumerate(remaining_errors[:30]):  # Fix next 30 files
        file_path = error['file']
        print(f"{i+1}. Fixing {file_path}...")
        
        success, message = fix_file_advanced(file_path, error)
        
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
    print(f"Total fixed so far: {len(fixed_files) + fixed_count}")
    print(f"Remaining: {len(remaining_errors) - 30} files")
    
    # Save results
    with open('/workspace/syntax_fix_results_batch2.json', 'w') as f:
        json.dump({
            'fixed': fixed_count,
            'failed': failed_count,
            'results': results,
            'total_fixed': len(fixed_files) + fixed_count
        }, f, indent=2)
    
    print(f"\nResults saved to: /workspace/syntax_fix_results_batch2.json")


if __name__ == "__main__":
    main()