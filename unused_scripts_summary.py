#!/usr/bin/env python3
"""
Analyze and categorize unused scripts for potential removal.
"""

import json
import os
from collections import defaultdict

def categorize_unused_files():
    """Categorize unused files by type and purpose."""
    
    with open('/workspace/unused_scripts_analysis.json', 'r') as f:
        data = json.load(f)
    
    unused_files = data['unused_files']
    
    categories = {
        'test_files': [],
        'demo_files': [],
        'utility_scripts': [],
        'fix_scripts': [],
        'validation_scripts': [],
        'analysis_scripts': [],
        'integration_scripts': [],
        'other': []
    }
    
    for file_path in unused_files:
        filename = os.path.basename(file_path)
        
        if filename.startswith('test_') or 'test' in filename.lower():
            categories['test_files'].append(file_path)
        elif filename.startswith('demo_') or 'demo' in filename.lower():
            categories['demo_files'].append(file_path)
        elif any(word in filename.lower() for word in ['fix', 'repair', 'correct']):
            categories['fix_scripts'].append(file_path)
        elif any(word in filename.lower() for word in ['validate', 'verify', 'check']):
            categories['validation_scripts'].append(file_path)
        elif any(word in filename.lower() for word in ['analyze', 'analysis', 'scan']):
            categories['analysis_scripts'].append(file_path)
        elif any(word in filename.lower() for word in ['integrate', 'merge', 'complete']):
            categories['integration_scripts'].append(file_path)
        elif any(word in filename.lower() for word in ['format', 'standardize', 'consolidate']):
            categories['utility_scripts'].append(file_path)
        else:
            categories['other'].append(file_path)
    
    return categories

def main():
    categories = categorize_unused_files()
    
    print("🔍 UNUSED SCRIPTS CATEGORIZATION")
    print("=" * 60)
    
    total_unused = sum(len(files) for files in categories.values())
    
    for category, files in categories.items():
        if files:
            print(f"\n📁 {category.upper().replace('_', ' ')} ({len(files)} files):")
            for i, file_path in enumerate(files[:10], 1):  # Show first 10
                print(f"  {i:2d}. {file_path}")
            if len(files) > 10:
                print(f"     ... and {len(files) - 10} more")
    
    print(f"\n📊 SUMMARY:")
    print(f"Total unused files: {total_unused}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"✅ SAFE TO REMOVE:")
    print(f"  - Test files: {len(categories['test_files'])} files")
    print(f"  - Demo files: {len(categories['demo_files'])} files")
    print(f"  - Fix scripts: {len(categories['fix_scripts'])} files (one-time use)")
    
    print(f"\n⚠️  REVIEW BEFORE REMOVING:")
    print(f"  - Validation scripts: {len(categories['validation_scripts'])} files")
    print(f"  - Analysis scripts: {len(categories['analysis_scripts'])} files")
    print(f"  - Utility scripts: {len(categories['utility_scripts'])} files")
    
    print(f"\n🔍 INVESTIGATE FURTHER:")
    print(f"  - Integration scripts: {len(categories['integration_scripts'])} files")
    print(f"  - Other files: {len(categories['other'])} files")

if __name__ == "__main__":
    main()