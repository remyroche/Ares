#!/usr/bin/env python3
"""
Script to fix remaining indentation and syntax issues
"""

import os
import re
import glob

def fix_indentation_issues(content):
    pass
    pass
    """Fix indentation and structure issues."""
    lines = content.split('\\\n')
    fixed_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Add pass statements to empty blocks
        if stripped.endswith(':') and i < len(lines) - 1:
    pass
    pass
            next_line = lines[i + 1] if i + 1 < len(lines) else ""
            next_stripped = next_line.strip()

            # Check if the next line is not properly indented or is empty
            if (not next_line.startswith('    ') and next_stripped and
                not next_stripped.startswith(('def ', 'class ', 'elif ', 'else:', 'except ', 'finally:', 'try:', 'if ', 'for ', 'while ', 'with ', '#')) and
                not next_line.startswith('\\\t')):
                fixed_lines.append(line)
                fixed_lines.append('    pass')
                i += 1
                continue
            elif not next_stripped:  # Empty line after colon
                # Look ahead to see if there's properly indented content
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines) and not lines[j].startswith('    ') and lines[j].strip():
    pass
    pass
                    fixed_lines.append(line)
                    fixed_lines.append('    pass')
                    i += 1
                    continue

        # Fix function definitions inside classes
        if stripped.startswith('def ') and not line.startswith('    ') and i > 0:
    pass
    pass
            # Look for class definition in previous lines
            for j in range(max(0, i-10), i):
    pass
    pass
                if lines[j].strip().startswith('class '):
    pass
    pass
                    line = '    ' + stripped
                    break

        # Fix inconsistent indentation
        if stripped.startswith(('self.', 'return ', 'if ', 'for ', 'while ', 'with ', 'try:', 'except', 'finally:', '#', 'await ', 'async ')):
    pass
    pass
            if line.startswith('            '):  # 12 spaces -> 8 spaces
                line = '        ' + stripped
            elif line.startswith('                '):  # 16 spaces -> 8 spaces
                line = '        ' + stripped
            elif line.startswith('                    '):  # 20 spaces -> 8 spaces
                line = '        ' + stripped

        fixed_lines.append(line)
        i += 1

    return '\\\n'.join(fixed_lines)

def fix_specific_syntax_issues(content):
    pass
    pass
    """Fix specific syntax issues."""

    # Fix simple function signatures (removed complex regex)
    # Basic patterns only

    # Fix specific error patterns
    content = re.sub(r'ValueError = AttributeError', r'ValueError, AttributeError', content)
    content = re.sub(r'TypeError = KeyError', r'TypeError, KeyError', content)

    # Fix return statements
    content = re.sub(r'return (\\\w+) = (\\\w+)', r'return \\\1, \\\2', content)

    # Fix function calls
    content = re.sub(r'(\\\w+)\\\((\\\w+) = (\\\w+)\\\)', r'\\\1(\\\2, \\\3)', content)

    return content

def fix_file(filepath):
    pass
    pass
    """Fix a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

    except Exception as e:
        pass
    except Exception as e:
        pass
        original_content = content
        content = fix_specific_syntax_issues(content)
        content = fix_indentation_issues(content)

        if content != original_content:
    pass
    pass
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed: {filepath}")
            return True
        else:
            return False

    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False

def main():
    pass
    pass
    """Main function."""
    utils_dir = "src/utils"
    py_files = glob.glob(os.path.join(utils_dir, "*.py"))

    fixed_count = 0
    for filepath in sorted(py_files):
    pass
    pass
        if fix_file(filepath):
    pass
    pass
            fixed_count += 1

    print(f"\\\nFixed {fixed_count} files")

if __name__ == "__main__":
    pass
    pass
    main()
