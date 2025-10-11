#!/usr/bin/env python3
"""
Consolidation Script for Rolling Operations

This script replaces duplicated rolling operation implementations
with calls to centralized methods from the base class.
"""

import os
import re
from pathlib import Path

def consolidate_rolling_operations(file_path: str) -> None:
    """Replace duplicated rolling operations with centralized methods."""
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace _rolling_mean_vectorized with _vectorbt_rolling_operation
    content = re.sub(
        r'def _rolling_mean_vectorized\(self, data: np\.ndarray, window: int\) -> np\.ndarray:\s*\n.*?return.*?\n',
        'def _rolling_mean_vectorized(self, data: np.ndarray, window: int) -> np.ndarray:\n        """Calculate rolling mean using centralized method."""\n        series = pd.Series(data)\n        return self._vectorbt_rolling_operation(series, "mean", window).values\n',
        content,
        flags=re.MULTILINE | re.DOTALL
    )
    
    # Replace _rolling_std_vectorized with _vectorbt_rolling_operation
    content = re.sub(
        r'def _rolling_std_vectorized\(self, data: np\.ndarray, window: int\) -> np\.ndarray:\s*\n.*?return.*?\n',
        'def _rolling_std_vectorized(self, data: np.ndarray, window: int) -> np.ndarray:\n        """Calculate rolling std using centralized method."""\n        series = pd.Series(data)\n        return self._vectorbt_rolling_operation(series, "std", window).values\n',
        content,
        flags=re.MULTILINE | re.DOTALL
    )
    
    # Replace _rolling_min_vectorized with _vectorbt_rolling_operation
    content = re.sub(
        r'def _rolling_min_vectorized\(self, data: np\.ndarray, window: int\) -> np\.ndarray:\s*\n.*?return.*?\n',
        'def _rolling_min_vectorized(self, data: np.ndarray, window: int) -> np.ndarray:\n        """Calculate rolling min using centralized method."""\n        series = pd.Series(data)\n        return self._vectorbt_rolling_operation(series, "min", window).values\n',
        content,
        flags=re.MULTILINE | re.DOTALL
    )
    
    # Replace _rolling_max_vectorized with _vectorbt_rolling_operation
    content = re.sub(
        r'def _rolling_max_vectorized\(self, data: np\.ndarray, window: int\) -> np\.ndarray:\s*\n.*?return.*?\n',
        'def _rolling_max_vectorized(self, data: np.ndarray, window: int) -> np.ndarray:\n        """Calculate rolling max using centralized method."""\n        series = pd.Series(data)\n        return self._vectorbt_rolling_operation(series, "max", window).values\n',
        content,
        flags=re.MULTILINE | re.DOTALL
    )
    
    # Replace _calculate_ema_vectorized with centralized method
    content = re.sub(
        r'def _calculate_ema_vectorized\(self, prices: np\.ndarray, span: int\) -> np\.ndarray:\s*\n.*?return.*?\n',
        'def _calculate_ema_vectorized(self, prices: np.ndarray, span: int) -> np.ndarray:\n        """Calculate EMA using centralized method."""\n        series = pd.Series(prices)\n        return self._calculate_ema_vectorized(series, span).values\n',
        content,
        flags=re.MULTILINE | re.DOTALL
    )
    
    # Write back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Processed rolling operations in {file_path}")

def main():
    """Main consolidation function."""
    
    # Files to process
    feature_files = [
        "src/feature_generation/categories/legacy.py"
    ]
    
    for file_path in feature_files:
        if os.path.exists(file_path):
            try:
                consolidate_rolling_operations(file_path)
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")
        else:
            print(f"⚠️ File not found: {file_path}")

if __name__ == "__main__":
    main()