#!/usr/bin/env python3
import sys

try:
    with open(sys.argv[1], 'r') as f:
        compile(f.read(), sys.argv[1], 'exec')
    print(f"✓ {sys.argv[1]} syntax is valid")
except SyntaxError as e:
    print(f"✗ {sys.argv[1]} has syntax error:")
    print(f"  Line {e.lineno}: {e.msg}")
    if e.text:
        print(f"  Text: {e.text.strip()}")