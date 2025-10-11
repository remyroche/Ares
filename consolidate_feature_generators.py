#!/usr/bin/env python3
"""
Consolidation Script for Feature Generators

This script removes redundant methods from feature generator subclasses
that are already implemented in the VectorizedFeatureGenerator base class.
"""

import os
import re
from pathlib import Path

def remove_redundant_methods(file_path: str) -> None:
    """Remove redundant methods from a feature generator file."""
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match the redundant methods
    pattern = r'\s*def optimize_dataframe_processing\(self, data: pd\.DataFrame\) -> pd\.DataFrame:\s*\n\s*"""Optimize DataFrame for vectorized processing\."""\s*\n\s*if hasattr\(self, \'vectorization_optimizer\'\) and self\.vectorization_optimizer:\s*\n\s*return self\.vectorization_optimizer\.optimize_dataframe_processing\(data\)\s*\n\s*return data\s*\n\s*\n\s*def vectorized_rolling_operations\(self, data: pd\.DataFrame, operations: List\[str\],\s*\n\s*windows: List\[int\], columns: Optional\[List\[str\]\] = None\) -> pd\.DataFrame:\s*\n\s*"""Perform vectorized rolling operations with hardware optimization\."""\s*\n\s*if hasattr\(self, \'vectorization_optimizer\'\) and self\.vectorization_optimizer:\s*\n\s*return self\.vectorization_optimizer\.vectorized_rolling_operations\(\s*\n\s*data, operations, windows, columns\s*\n\s*\)\s*\n\s*return data\s*\n'
    
    # Remove the redundant methods
    new_content = re.sub(pattern, '', content, flags=re.MULTILINE | re.DOTALL)
    
    # Write back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"✅ Processed {file_path}")

def main():
    """Main consolidation function."""
    
    # Files to process
    feature_files = [
        "src/feature_generation/categories/momentum.py",
        "src/feature_generation/categories/trend.py", 
        "src/feature_generation/categories/oscillator.py",
        "src/feature_generation/categories/legacy.py",
        "src/feature_generation/categories/normalization.py"
    ]
    
    for file_path in feature_files:
        if os.path.exists(file_path):
            try:
                remove_redundant_methods(file_path)
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")
        else:
            print(f"⚠️ File not found: {file_path}")

if __name__ == "__main__":
    main()