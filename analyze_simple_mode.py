#!/usr/bin/env python3
"""Find and remove simple mode configuration from compare_tbm_parameters.py"""

import re

file_path = '/Users/remyroche/Documents/Ares/extreme_price_movements/offline_optimisers/compare_tbm_parameters.py'

with open(file_path, 'r') as f:
    content = f.read()

# Look for simple-related patterns
simple_patterns = [
    r'simple_tight_\w+',
    r'simple_wide_\w+',
    r'simple_generated',
    r'SIMPLE_\w+',
    r'"simple"',
    r"'simple'",
]

found_patterns = {}
for pattern in simple_patterns:
    matches = re.findall(pattern, content)
    if matches:
        found_patterns[pattern] = matches[:5]  # First 5 matches

print("Found simple patterns:")
for pattern, matches in found_patterns.items():
    print(f"  {pattern}: {matches}")

# Look for hardcoded metric values (0.55, 0.52, 0.05, 1.0, 0.15, 0.2, 0.3, 0.6, 0.5)
metric_pattern = r'"cell_auc".*0\.55|"cell_bind".*0\.3|"cell_ece".*0\.15|"cell_brier".*0\.2'
metric_matches = re.findall(r'cell_auc.*?(0\.\d+)', content)
print(f"\nCell AUC values found: {list(set(metric_matches))[:10] if metric_matches else ['None']}")

# Check if there's a simple mode flag or condition
flag_patterns = [
    r'use_simple\s*=\s*(True|False)',
    r'simple_mode\s*=\s*(True|False)',
    r'if.*simple',
    r'def.*simple',
]

print("\nChecking for simple mode flags:")
for pattern in flag_patterns:
    matches = re.findall(pattern, content, re.IGNORECASE)
    if matches:
        print(f"  {pattern}: {matches[:3]}")
