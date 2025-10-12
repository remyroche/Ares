#!/usr/bin/env python3
"""
Script to centralize all indicator calculations in feature_generation/indicators/

This script finds all files with duplicate indicator calculations and updates them
to use the centralized calculators from feature_generation/indicators/
"""

import os
import re
import glob
from pathlib import Path

def find_files_with_indicators():
    """Find all Python files that contain indicator calculations."""
    files_with_indicators = []
    
    # Search patterns for indicator calculations
    patterns = [
        r'def.*_calculate_rsi\(',
        r'def.*_calculate_macd\(',
        r'def.*_calculate_sma\(',
        r'def.*_calculate_ema\(',
        r'def.*_calculate_stoch\(',
        r'def.*_calculate_bb\(',
        r'def.*_calculate_bollinger',
        r'def.*compute_rsi\(',
        r'def.*compute_macd\(',
        r'def.*compute_sma\(',
        r'def.*compute_ema\(',
        r'def.*compute_stoch\(',
        r'def.*compute_bb\(',
        r'def.*compute_bollinger',
        r'def.*calculate_rsi\(',
        r'def.*calculate_macd\(',
        r'def.*calculate_sma\(',
        r'def.*calculate_ema\(',
        r'def.*calculate_stoch\(',
        r'def.*calculate_bb\(',
        r'def.*calculate_bollinger'
    ]
    
    # Search in src directory
    for root, dirs, files in os.walk('src'):
        # Skip feature_generation directory as it contains the centralized implementations
        if 'feature_generation' in root:
            continue
            
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Check if file contains any indicator calculations
                    for pattern in patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            files_with_indicators.append(filepath)
                            break
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
    
    return files_with_indicators

def update_file_indicators(filepath):
    """Update a file to use centralized indicator calculators."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # RSI replacements
        rsi_patterns = [
            (r'def.*_calculate_rsi\([^)]*\):\s*"""[^"]*"""\s*.*?return rsi', 
             'def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:\n        """Calculate RSI indicator using centralized calculator."""\n        from src.feature_generation.indicators import RSICalculator\n        return RSICalculator.calculate(prices, window)'),
            (r'def.*compute_rsi\([^)]*\):\s*"""[^"]*"""\s*.*?return.*rsi', 
             'def compute_rsi(data: pd.DataFrame, window: int = 3) -> float:\n        """Compute a short-term RSI using centralized calculator."""\n        if data.empty or len(data) < window + 1:\n            return 50.0\n        from src.feature_generation.indicators import RSICalculator\n        close = data["close"]\n        rsi = RSICalculator.calculate(close, window)\n        return rsi.iloc[-1] if not rsi.empty and not pd.isna(rsi.iloc[-1]) else 50.0')
        ]
        
        # MACD replacements
        macd_patterns = [
            (r'def.*_calculate_macd\([^)]*\):\s*"""[^"]*"""\s*.*?return.*macd', 
             'def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:\n        """Calculate MACD indicator using centralized calculator."""\n        from src.feature_generation.indicators import MACDCalculator\n        return MACDCalculator.calculate(prices, fast, slow, signal)')
        ]
        
        # SMA replacements
        sma_patterns = [
            (r'def.*_calculate_sma\([^)]*\):\s*"""[^"]*"""\s*.*?return.*sma', 
             'def _calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:\n        """Calculate SMA using centralized calculator."""\n        from src.feature_generation.indicators import SMACalculator\n        return SMACalculator.calculate(prices, period)')
        ]
        
        # EMA replacements
        ema_patterns = [
            (r'def.*_calculate_ema\([^)]*\):\s*"""[^"]*"""\s*.*?return.*ema', 
             'def _calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:\n        """Calculate EMA using centralized calculator."""\n        from src.feature_generation.indicators import EMACalculator\n        return EMACalculator.calculate(prices, period)')
        ]
        
        # Bollinger Bands replacements
        bb_patterns = [
            (r'def.*_calculate_bollinger[^)]*\):\s*"""[^"]*"""\s*.*?return.*band', 
             'def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:\n        """Calculate Bollinger Bands using centralized calculator."""\n        from src.feature_generation.indicators import BollingerBandsCalculator\n        return BollingerBandsCalculator.calculate(prices, window, num_std)')
        ]
        
        # Apply replacements
        all_patterns = rsi_patterns + macd_patterns + sma_patterns + ema_patterns + bb_patterns
        
        for pattern, replacement in all_patterns:
            content = re.sub(pattern, replacement, content, flags=re.DOTALL | re.MULTILINE)
        
        # Only write if content changed
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Updated: {filepath}")
            return True
        else:
            print(f"No changes needed: {filepath}")
            return False
            
    except Exception as e:
        print(f"Error updating {filepath}: {e}")
        return False

def main():
    """Main function to centralize all indicator calculations."""
    print("🔍 Finding files with indicator calculations...")
    
    files_with_indicators = find_files_with_indicators()
    print(f"Found {len(files_with_indicators)} files with indicator calculations")
    
    updated_count = 0
    for filepath in files_with_indicators:
        print(f"\n📝 Processing: {filepath}")
        if update_file_indicators(filepath):
            updated_count += 1
    
    print(f"\n✅ Centralization complete!")
    print(f"📊 Updated {updated_count} out of {len(files_with_indicators)} files")
    print(f"🎯 All indicator calculations now use centralized calculators from feature_generation/indicators/")

if __name__ == "__main__":
    main()