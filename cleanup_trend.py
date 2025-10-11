#!/usr/bin/env python3
"""
Cleanup script for trend.py to remove duplicate ADX implementation
"""

import re

def cleanup_trend_file():
    """Remove duplicate ADX implementation from trend.py"""
    
    with open('src/feature_generation/categories/trend.py', 'r') as f:
        content = f.read()
    
    # Remove the duplicate ADX implementation
    # Find the start of the ADX implementation and remove it
    pattern = r'        """Calculate Average Directional Index \(ADX\)\."""\s*\n.*?return adx\.values'
    
    # Replace with empty string
    content = re.sub(pattern, '', content, flags=re.MULTILINE | re.DOTALL)
    
    # Fix the class definition that got broken
    content = re.sub(r'return adx\.valuesclass DirectionalSignalGenerator', 
                    'class DirectionalSignalGenerator', content)
    
    # Fix the class definition that got broken
    content = re.sub(r'return directional_signalclass TrendScoreGenerator', 
                    'class TrendScoreGenerator', content)
    
    with open('src/feature_generation/categories/trend.py', 'w') as f:
        f.write(content)
    
    print("✅ Cleaned up trend.py")

if __name__ == "__main__":
    cleanup_trend_file()