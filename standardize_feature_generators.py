#!/usr/bin/env python3
"""
Standardization Script for Feature Generators

This script standardizes all feature generators to use centralized methods
from VectorizedFeatureGenerator base class.
"""

import os
import re
from pathlib import Path

def standardize_rolling_operations(file_path: str) -> None:
    """Standardize rolling operations to use centralized methods."""
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace direct pandas rolling operations with centralized methods
    replacements = [
        # Rolling mean
        (r'(\w+)\.rolling\(window=(\w+)\)\.mean\(\)', r'self._calculate_sma_vectorized(\1, \2)'),
        (r'(\w+)\.rolling\(window=(\w+)\)\.std\(\)', r'self._calculate_rolling_std_vectorized(\1, \2)'),
        (r'(\w+)\.rolling\(window=(\w+)\)\.min\(\)', r'self._calculate_rolling_min_vectorized(\1, \2)'),
        (r'(\w+)\.rolling\(window=(\w+)\)\.max\(\)', r'self._calculate_rolling_max_vectorized(\1, \2)'),
        (r'(\w+)\.rolling\(window=(\w+)\)\.sum\(\)', r'self._calculate_rolling_sum_vectorized(\1, \2)'),
        
        # EMA operations
        (r'(\w+)\.ewm\(span=(\w+)\)\.mean\(\)', r'self._calculate_ema_vectorized(\1, \2)'),
        (r'(\w+)\.ewm\(alpha=([\d.]+)\)\.mean\(\)', r'self._calculate_ema_vectorized(\1, int(2/\2-1))'),
        
        # Quantile operations
        (r'(\w+)\.rolling\(window=(\w+)\)\.quantile\(([\d.]+)\)', r'self._calculate_rolling_quantile_vectorized(\1, \2, \3)'),
    ]
    
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)
    
    # Replace custom rolling implementations with centralized methods
    custom_rolling_patterns = [
        # Custom rolling mean implementations
        (r'def _rolling_mean_vectorized\(self, data: np\.ndarray, window: int\) -> np\.ndarray:.*?return.*?\n', 
         'def _rolling_mean_vectorized(self, data: np.ndarray, window: int) -> np.ndarray:\n        """Calculate rolling mean using centralized method."""\n        series = pd.Series(data)\n        return self._calculate_sma_vectorized(series, window).values\n'),
        
        # Custom rolling std implementations
        (r'def _rolling_std_vectorized\(self, data: np\.ndarray, window: int\) -> np\.ndarray:.*?return.*?\n',
         'def _rolling_std_vectorized(self, data: np.ndarray, window: int) -> np.ndarray:\n        """Calculate rolling std using centralized method."""\n        series = pd.Series(data)\n        return self._calculate_rolling_std_vectorized(series, window).values\n'),
        
        # Custom rolling min implementations
        (r'def _rolling_min_vectorized\(self, data: np\.ndarray, window: int\) -> np\.ndarray:.*?return.*?\n',
         'def _rolling_min_vectorized(self, data: np.ndarray, window: int) -> np.ndarray:\n        """Calculate rolling min using centralized method."""\n        series = pd.Series(data)\n        return self._calculate_rolling_min_vectorized(series, window).values\n'),
        
        # Custom rolling max implementations
        (r'def _rolling_max_vectorized\(self, data: np\.ndarray, window: int\) -> np\.ndarray:.*?return.*?\n',
         'def _rolling_max_vectorized(self, data: np.ndarray, window: int) -> np.ndarray:\n        """Calculate rolling max using centralized method."""\n        series = pd.Series(data)\n        return self._calculate_rolling_max_vectorized(series, window).values\n'),
        
        # Custom EMA implementations
        (r'def _calculate_ema_vectorized\(self, prices: np\.ndarray, span: int\) -> np\.ndarray:.*?return.*?\n',
         'def _calculate_ema_vectorized(self, prices: np.ndarray, span: int) -> np.ndarray:\n        """Calculate EMA using centralized method."""\n        series = pd.Series(prices)\n        return self._calculate_ema_vectorized(series, span).values\n'),
    ]
    
    for pattern, replacement in custom_rolling_patterns:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
    
    # Write back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Standardized rolling operations in {file_path}")

def standardize_vectorbt_usage(file_path: str) -> None:
    """Standardize VectorBT usage patterns."""
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace direct VectorBT calls with centralized methods
    vectorbt_replacements = [
        # Replace vbt.RSI.run() calls
        (r'vbt\.RSI\.run\((\w+), window=(\w+)\)', r'self._calculate_rsi_vectorized(\1, \2)'),
        (r'vbt\.MACD\.run\((\w+), fast_window=(\w+), slow_window=(\w+), signal_window=(\w+)\)', 
         r'self._calculate_macd_vectorized(\1, \2, \3, \4)'),
        (r'vbt\.BBANDS\.run\((\w+), window=(\w+), alpha=(\w+)\)', 
         r'self._calculate_bollinger_bands_vectorized(\1, \2, \3)'),
        (r'vbt\.ATR\.run\((\w+), (\w+), (\w+), window=(\w+)\)', 
         r'self._calculate_atr_vectorized(\1, \2, \3, \4)'),
        (r'vbt\.STOCH\.run\((\w+), (\w+), (\w+), k_window=(\w+), d_window=(\w+)\)', 
         r'self._calculate_stochastic_vectorized(\1, \2, \3, \4, \5)'),
        (r'vbt\.WILLR\.run\((\w+), (\w+), (\w+), window=(\w+)\)', 
         r'self._calculate_williams_r_vectorized(\1, \2, \3, \4)'),
        (r'vbt\.OBV\.run\((\w+), (\w+)\)', 
         r'self._calculate_obv_vectorized(\1, \2)'),
    ]
    
    for pattern, replacement in vectorbt_replacements:
        content = re.sub(pattern, replacement, content)
    
    # Write back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Standardized VectorBT usage in {file_path}")

def main():
    """Main standardization function."""
    
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
                print(f"🔄 Processing {file_path}...")
                standardize_rolling_operations(file_path)
                standardize_vectorbt_usage(file_path)
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")
        else:
            print(f"⚠️ File not found: {file_path}")

if __name__ == "__main__":
    main()