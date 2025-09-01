#!/usr/bin/env python3
"""
Fix indentation in adaptive_trial_allocator.py
"""

def fix_file():
    """Fix the indentation issues in the file."""
    with open('src/training/optimization/adaptive_trial_allocator.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix the problematic patterns
    # Pattern 1: Fix try-except blocks with pass statements
    content = content.replace(
        '        try:\n    pass  # TODO: Add proper exception handling\nexcept Exception as e:\n    pass  # TODO: Add proper exception handling',
        '        try:'
    )
    
    # Pattern 2: Fix try-except blocks with pass statements (different indentation)
    content = content.replace(
        '        try:\n            pass  # TODO: Add proper exception handling\n        except Exception as e:\n            pass  # TODO: Add proper exception handling',
        '        try:'
    )
    
    # Pattern 3: Fix try-except blocks with pass statements (no indentation)
    content = content.replace(
        'try:\n    pass  # TODO: Add proper exception handling\nexcept Exception as e:\n    pass  # TODO: Add proper exception handling',
        'try:'
    )
    
    with open('src/training/optimization/adaptive_trial_allocator.py', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    fix_file()
    print("Fixed adaptive_trial_allocator.py")