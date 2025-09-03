#!/usr/bin/env python3
"""Fix the tactician.py file specifically."""

def fix_tactician():
    with open("/workspace/src/tactician/tactician.py", 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    fixed_lines = []
    
    in_init = False
    init_indent = 0
    
    for i, line in enumerate(lines):
        # Track when we're in __init__
        if 'def __init__(' in line:
            in_init = True
            init_indent = len(line) - len(line.lstrip())
        
        # Fix lines that should be indented inside __init__
        if in_init and i >= 35 and i <= 150:
            stripped = line.strip()
            if stripped and not line.startswith(' ' * (init_indent + 8)):
                # This line needs proper indentation
                if any(stripped.startswith(x) for x in ['#', 'self.', 'step17_config', 'tactician_config']):
                    line = ' ' * (init_indent + 8) + stripped
        
        # Check if we've exited __init__
        if in_init and line.strip() and not line.startswith(' '):
            in_init = False
            
        fixed_lines.append(line)
    
    with open("/workspace/src/tactician/tactician.py", 'w') as f:
        f.write('\n'.join(fixed_lines))
    
    print("Fixed tactician.py")

if __name__ == "__main__":
    fix_tactician()