#!/usr/bin/env python3
"""
Comprehensive Fix Script: Update all feature_engineering references

This script finds and fixes ALL references to feature_engineering across the codebase
to point to the new feature_generation.utils location.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple

def read_file_safe(file_path: Path) -> str:
    """Safely read file content."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return ""

def write_file_safe(file_path: Path, content: str) -> bool:
    """Safely write file content."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"❌ Error writing {file_path}: {e}")
        return False

def get_replacement_patterns() -> List[Tuple[str, str]]:
    """Get all replacement patterns for feature_engineering references."""
    return [
        # Direct imports
        ('from src.feature_engineering', 'from src.feature_generation.utils'),
        ('import src.feature_engineering', 'import src.feature_generation.utils'),
        
        # Relative imports (for files within the moved structure)
        ('from ..feature_engineering', 'from .'),
        ('from ...feature_engineering', 'from ..'),
        ('from ....feature_engineering', 'from ...'),
        
        # Module path references
        ('src.feature_engineering.', 'src.feature_generation.utils.'),
        ('"src.feature_engineering.', '"src.feature_generation.utils.'),
        ("'src.feature_engineering.", "'src.feature_generation.utils."),
        
        # Config and string references
        ('feature_engineering/', 'feature_generation/utils/'),
        ('feature_engineering.', 'feature_generation.utils.'),
        
        # Specific common imports that might need special handling
        ('from src.feature_engineering.step06_utility_container', 'from src.feature_generation.utils.step06_utility_container'),
        ('from src.feature_engineering.step06_enhanced_feature_engineering', 'from src.feature_generation.utils.step06_enhanced_feature_engineering'),
        ('from src.feature_engineering.optimization', 'from src.feature_generation.utils.optimization'),
        ('from src.feature_engineering.math_validation', 'from src.feature_generation.utils.math_validation'),
        ('from src.feature_engineering.feature_generators', 'from src.feature_generation.utils.feature_generators'),
        ('from src.feature_engineering.cross_timeframe', 'from src.feature_generation.utils.cross_timeframe'),
        ('from src.feature_engineering.enhanced_', 'from src.feature_generation.utils.enhanced_'),
        ('from src.feature_engineering.fractional_', 'from src.feature_generation.utils.fractional_'),
        ('from src.feature_engineering.sr_', 'from src.feature_generation.utils.sr_'),
        
        # Import aliases and specific patterns
        ('FeatureEngineering', 'FeatureEngineering'),  # Keep class names the same
        ('feature_engineering_config', 'feature_engineering_config'),  # Keep variable names
    ]

def fix_file_references(file_path: Path) -> bool:
    """Fix all feature_engineering references in a single file."""
    content = read_file_safe(file_path)
    if not content:
        return False
    
    original_content = content
    replacements = get_replacement_patterns()
    
    # Apply replacements
    for old_pattern, new_pattern in replacements:
        content = content.replace(old_pattern, new_pattern)
    
    # Special handling for files that are within the utils directory
    if '/utils/' in str(file_path):
        # Fix internal references within utils
        content = content.replace('from src.feature_generation.utils.', 'from .')
        content = content.replace('from ...feature_generation.utils', 'from ..')
        content = content.replace('from ..feature_generation.utils', 'from .')
    
    if content != original_content:
        if write_file_safe(file_path, content):
            return True
    
    return False

def find_all_files_with_references() -> List[Path]:
    """Find all files that contain feature_engineering references."""
    print("🔍 Scanning for feature_engineering references...")
    
    files_with_refs = []
    
    # Search in key directories
    search_dirs = [
        "src/training",
        "src/utils", 
        "src/analyst",
        "src/trading",
        "src/feature_generation",
        "src/config",
        "src/launcher",
        "src/steps",
        "src/core",
        "src/custom_types",
        "src/tactician"
    ]
    
    for search_dir in search_dirs:
        search_path = Path(search_dir)
        if search_path.exists():
            for py_file in search_path.rglob("*.py"):
                # Skip backup files
                if 'backup' in str(py_file):
                    continue
                
                content = read_file_safe(py_file)
                if 'feature_engineering' in content:
                    files_with_refs.append(py_file)
            
            # Also check YAML and MD files
            for ext in ["*.yaml", "*.yml", "*.md"]:
                for config_file in search_path.rglob(ext):
                    if 'backup' in str(config_file):
                        continue
                    
                    content = read_file_safe(config_file)
                    if 'feature_engineering' in content:
                        files_with_refs.append(config_file)
    
    print(f"📊 Found {len(files_with_refs)} files with feature_engineering references")
    return files_with_refs

def categorize_files(files: List[Path]) -> Dict[str, List[Path]]:
    """Categorize files by type for better processing."""
    categories = {
        'python': [],
        'config': [],
        'docs': [],
        'utils_internal': []
    }
    
    for file_path in files:
        if '/utils/' in str(file_path) and file_path.suffix == '.py':
            categories['utils_internal'].append(file_path)
        elif file_path.suffix == '.py':
            categories['python'].append(file_path)
        elif file_path.suffix in ['.yaml', '.yml']:
            categories['config'].append(file_path)
        elif file_path.suffix in ['.md', '.txt']:
            categories['docs'].append(file_path)
    
    return categories

def fix_specific_known_issues():
    """Fix specific known problematic files."""
    print("\n🎯 Fixing specific known issues...")
    
    specific_fixes = [
        {
            'file': 'src/training/steps/market_analysis/enhanced_market_analysis_orchestrator.py',
            'old': 'from src.feature_engineering.step06_enhanced_feature_engineering import EnhancedFeatureEngineering as FeatureEngineeringStep',
            'new': 'from src.feature_generation.utils.step06_enhanced_feature_engineering import EnhancedFeatureEngineering as FeatureEngineeringStep'
        },
        {
            'file': 'src/feature_generation/core/feature_bank.py',
            'old': 'from ...feature_engineering.feature_generation_optimization import',
            'new': 'from ..utils.feature_generation_optimization import'
        }
    ]
    
    fixed_count = 0
    for fix in specific_fixes:
        file_path = Path(fix['file'])
        if file_path.exists():
            content = read_file_safe(file_path)
            if fix['old'] in content:
                content = content.replace(fix['old'], fix['new'])
                if write_file_safe(file_path, content):
                    print(f"  ✅ Fixed specific issue in {file_path.name}")
                    fixed_count += 1
    
    print(f"✅ Fixed {fixed_count} specific issues")

def main():
    """Main function to fix all references."""
    print("🚀 Starting Comprehensive Feature Engineering Reference Fix")
    print("=" * 60)
    
    # Step 1: Find all files with references
    files_with_refs = find_all_files_with_references()
    
    if not files_with_refs:
        print("✅ No files with feature_engineering references found")
        return
    
    # Step 2: Categorize files
    categories = categorize_files(files_with_refs)
    
    print(f"\n📋 File categories:")
    for category, file_list in categories.items():
        print(f"  {category}: {len(file_list)} files")
    
    # Step 3: Fix files by category
    total_fixed = 0
    
    # Fix Python files first
    print(f"\n🔧 Fixing Python files...")
    for file_path in categories['python']:
        if fix_file_references(file_path):
            print(f"  ✅ Fixed {file_path.relative_to(Path.cwd())}")
            total_fixed += 1
    
    # Fix utils internal files (need special handling)
    print(f"\n🔧 Fixing utils internal files...")
    for file_path in categories['utils_internal']:
        if fix_file_references(file_path):
            print(f"  ✅ Fixed {file_path.relative_to(Path.cwd())}")
            total_fixed += 1
    
    # Fix config files
    print(f"\n🔧 Fixing config files...")
    for file_path in categories['config']:
        if fix_file_references(file_path):
            print(f"  ✅ Fixed {file_path.relative_to(Path.cwd())}")
            total_fixed += 1
    
    # Fix documentation
    print(f"\n🔧 Fixing documentation...")
    for file_path in categories['docs']:
        if fix_file_references(file_path):
            print(f"  ✅ Fixed {file_path.relative_to(Path.cwd())}")
            total_fixed += 1
    
    # Step 4: Fix specific known issues
    fix_specific_known_issues()
    
    # Step 5: Summary
    print(f"\n🎉 Reference fix completed!")
    print(f"📊 Fixed {total_fixed} files total")
    
    # Step 6: Verification
    print(f"\n🔍 Verifying fixes...")
    remaining_files = find_all_files_with_references()
    remaining_files = [f for f in remaining_files if 'backup' not in str(f)]
    
    if remaining_files:
        print(f"⚠️ {len(remaining_files)} files still have references:")
        for file_path in remaining_files[:10]:  # Show first 10
            print(f"  - {file_path.relative_to(Path.cwd())}")
        if len(remaining_files) > 10:
            print(f"  ... and {len(remaining_files) - 10} more")
    else:
        print("✅ All references fixed!")
    
    return total_fixed

if __name__ == "__main__":
    main()