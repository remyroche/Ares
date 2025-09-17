#!/usr/bin/env python3
"""
Simple Fix Script: Update feature_engineering references
"""

import os
import re
from pathlib import Path
from typing import List, Tuple

def fix_file(file_path: Path) -> bool:
    """Fix feature_engineering references in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Define replacement patterns
        replacements = [
            ('from src.feature_engineering', 'from src.feature_generation.utils'),
            ('import src.feature_engineering', 'import src.feature_generation.utils'),
            ('src.feature_engineering.', 'src.feature_generation.utils.'),
            ('"src.feature_engineering.', '"src.feature_generation.utils.'),
            ("'src.feature_engineering.", "'src.feature_generation.utils."),
        ]
        
        # Apply replacements
        for old, new in replacements:
            content = content.replace(old, new)
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main function."""
    print("🔧 Fixing feature_engineering references...")
    
    # Find Python files with references
    fixed_count = 0
    
    for root, dirs, files in os.walk("src"):
        # Skip backup directories
        if 'backup' in root:
            continue
            
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if 'feature_engineering' in content:
                        if fix_file(file_path):
                            print(f"✅ Fixed {file_path}")
                            fixed_count += 1
                
                except Exception as e:
                    print(f"❌ Error with {file_path}: {e}")
    
    print(f"\n🎉 Fixed {fixed_count} files")

if __name__ == "__main__":
    main()