#!/usr/bin/env python3
"""Find the exact location of unmatched triple quotes"""

with open('src/training/steps/market_analysis/clusters/cluster_quality_assessor.py', 'r') as f:
    lines = f.readlines()

stack = []
for i, line in enumerate(lines, 1):
    # Count triple quotes in this line
    pos = 0
    while True:
        idx = line.find('"""', pos)
        if idx == -1:
            break
        if len(stack) % 2 == 0:
            # Opening triple quote
            stack.append((i, idx))
        else:
            # Closing triple quote
            if stack:
                opening = stack.pop()
                print(f"Closed triple quote: opened at line {opening[0]}, closed at line {i}")
        pos = idx + 3

if stack:
    print(f"❌ Unmatched triple quote at line {stack[-1][0]}")
    print(f"Line content: {lines[stack[-1][0]-1].strip()}")
else:
    print("✅ All triple quotes are matched")
