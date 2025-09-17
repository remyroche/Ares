#!/usr/bin/env python3
"""
Targeted Import Fix Script

This script fixes imports more carefully after the consolidation.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple

def find_files_with_feature_engineering_imports():
    """Find all files that import from feature_engineering."""
    print("🔍 Finding files with feature_engineering imports...")
    
    files_with_imports = []
    
    # Search in specific directories
    search_dirs = [
        "src/training",
        "src/utils", 
        "src/analyst",
        "src/trading",
        "src/feature_generation",
        "src/examples"
    ]
    
    for search_dir in search_dirs:
        if Path(search_dir).exists():
            for py_file in Path(search_dir).rglob("*.py"):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if 'feature_engineering' in content:
                        files_with_imports.append(py_file)
                        
                except Exception as e:
                    print(f"⚠️ Error reading {py_file}: {e}")
    
    print(f"📊 Found {len(files_with_imports)} files with feature_engineering imports")
    return files_with_imports

def update_file_imports(file_path: Path) -> bool:
    """Update imports in a specific file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Define specific import patterns to replace
        replacements = [
            # Direct imports
            ('from src.feature_engineering', 'from src.feature_generation.utils'),
            ('import src.feature_engineering', 'import src.feature_generation.utils'),
            
            # Relative imports from within the moved code
            ('from ...feature_engineering', 'from ..utils'),
            ('from ..feature_engineering', 'from .utils'),
            ('from .feature_engineering', 'from .utils'),
            
            # Module references
            ('src.feature_engineering.', 'src.feature_generation.utils.'),
            
            # Specific common imports
            ('from src.feature_engineering.optimization', 'from src.feature_generation.utils.optimization'),
            ('from src.feature_engineering.step06_', 'from src.feature_generation.utils.step06_'),
            ('from src.feature_engineering.cross_timeframe', 'from src.feature_generation.utils.cross_timeframe'),
            ('from src.feature_engineering.enhanced_', 'from src.feature_generation.utils.enhanced_'),
            ('from src.feature_engineering.feature_', 'from src.feature_generation.utils.feature_'),
            ('from src.feature_engineering.math_validation', 'from src.feature_generation.utils.math_validation'),
            ('from src.feature_engineering.sr_', 'from src.feature_generation.utils.sr_'),
        ]
        
        for old, new in replacements:
            content = content.replace(old, new)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Updated {file_path.relative_to(Path.cwd())}")
            return True
        
        return False
        
    except Exception as e:
        print(f"❌ Error updating {file_path}: {e}")
        return False

def fix_internal_utils_imports():
    """Fix imports within the utils directory itself."""
    print("\n🔧 Fixing internal utils imports...")
    
    utils_dir = Path("src/feature_generation/utils")
    if not utils_dir.exists():
        print(f"❌ Utils directory not found: {utils_dir}")
        return
    
    updated_files = 0
    
    for py_file in utils_dir.rglob("*.py"):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Fix internal imports within utils
            internal_replacements = [
                # Relative imports that need updating
                ('from ...feature_engineering', 'from ..'),
                ('from ..feature_engineering', 'from .'),
                ('from .feature_engineering', 'from .'),
                
                # Absolute imports within utils
                ('from src.feature_engineering.', 'from src.feature_generation.utils.'),
                ('import src.feature_engineering.', 'import src.feature_generation.utils.'),
                
                # Fix cross-references within utils
                ('src.feature_engineering.', 'src.feature_generation.utils.'),
            ]
            
            for old, new in internal_replacements:
                content = content.replace(old, new)
            
            if content != original_content:
                with open(py_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"  ✅ Fixed internal imports in {py_file.name}")
                updated_files += 1
                
        except Exception as e:
            print(f"  ❌ Error fixing {py_file}: {e}")
    
    print(f"✅ Fixed internal imports in {updated_files} files")

def update_specific_known_files():
    """Update specific files that we know need changes."""
    print("\n🎯 Updating specific known files...")
    
    specific_files = [
        # HMM compatibility
        "src/feature_generation/compatibility/hmm_compatibility.py",
        "src/hmm_feature_compatibility.py",
        
        # Training pipeline files
        "src/training/steps/market_analysis/feature_lookback_optimization/feature_lookback_optimization.py",
        "src/training/steps/market_analysis/enhanced_market_analysis_orchestrator.py",
        
        # Core feature generation
        "src/feature_generation/core/feature_bank.py",
        
        # Examples
        "src/examples/sr_feature_integration_example.py",
    ]
    
    updated_count = 0
    
    for file_path_str in specific_files:
        file_path = Path(file_path_str)
        if file_path.exists():
            if update_file_imports(file_path):
                updated_count += 1
        else:
            print(f"⚠️ File not found: {file_path}")
    
    print(f"✅ Updated {updated_count} specific files")

def main():
    """Main function to fix imports."""
    print("🚀 Starting Targeted Import Fix")
    print("=" * 40)
    
    # Step 1: Fix internal utils imports first
    fix_internal_utils_imports()
    
    # Step 2: Update specific known files
    update_specific_known_files()
    
    # Step 3: Find and update all files with feature_engineering imports
    files_to_update = find_files_with_feature_engineering_imports()
    updated_files = 0
    
    for file_path in files_to_update:
        if update_file_imports(file_path):
            updated_files += 1
    
    print(f"\n🎉 Import fix completed!")
    print(f"📊 Updated imports in {updated_files} files total")
    
    print("\n📋 Next: Test imports with a simple check...")
    return updated_files

if __name__ == "__main__":
    main()