#!/usr/bin/env python3
"""
Script to fix import patterns from relative to absolute imports
"""

import os
import re

def fix_imports_in_file(file_path):
    """Fix relative imports to absolute imports in a file"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        original_content = content

        # Fix vectorbt_rolling_optimizer imports
        content = re.sub(
            r'from \.\.utils\.vectorbt_rolling_optimizer import',
            r'from src.feature_generation.utils.vectorbt_rolling_optimizer import',
            content
        )

        # Fix unified_vectorization_manager imports
        content = re.sub(
            r'from \.\.utils\.unified_vectorization_manager import',
            r'from src.feature_generation.utils.unified_vectorization_manager import',
            content
        )

        # Fix other relative imports that might exist
        content = re.sub(
            r'from \.\.utils\.([^.]+) import',
            r'from src.feature_generation.utils.\1 import',
            content
        )

        # Fix triple-dot relative imports (from ...utils.)
        content = re.sub(
            r'from \.\.\.utils\.([^.]+) import',
            r'from src.feature_generation.utils.\1 import',
            content
        )

        if content != original_content:
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"Fixed imports in {file_path}")
            return True
        else:
            print(f"No changes needed in {file_path}")
            return False

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def fix_all_files():
    """Fix imports in all files that need it"""
    files_to_fix = [
        "src/feature_generation/categories/momentum.py",
        "src/feature_generation/categories/entropy.py",
        "src/feature_generation/categories/returns.py",
        "src/feature_generation/core/feature_generator.py",
        "src/feature_generation/categories/oscillator.py",
        "src/feature_generation/categories/candlestick_pattern.py",
        "src/feature_generation/categories/volatility.py",
        "src/feature_generation/categories/support_resistance.py",
        "src/feature_generation/categories/microstructure_features.py",
        "src/feature_generation/categories/cross_timeframe.py",
        "src/feature_generation/categories/regime_features.py",
        "src/feature_generation/categories/trend.py",
        "src/feature_generation/categories/interaction.py",
        "src/feature_generation/categories/time.py",
        "src/feature_generation/categories/vectorbt_acceleration.py",
        "src/feature_generation/categories/spectral_features.py",
        "src/feature_generation/categories/negative_learning.py",
        "src/feature_generation/categories/advanced_statistical.py"
    ]

    fixed_count = 0
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            if fix_imports_in_file(file_path):
                fixed_count += 1
        else:
            print(f"File not found: {file_path}")

    print(f"\nFixed imports in {fixed_count} files")

if __name__ == "__main__":
    fix_all_files()
