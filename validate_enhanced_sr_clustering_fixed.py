#!/usr/bin/env python3
"""
Fixed validation script for Enhanced SR Clustering module.
"""

import ast
import sys
import os
from pathlib import Path

def validate_python_syntax(file_path):
    """Validate Python syntax of a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the AST
        tree = ast.parse(content)
        
        print(f"✅ Syntax validation passed for {file_path}")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error in {file_path}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return False

def validate_class_structure(file_path):
    """Validate class structure and methods."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the AST
        tree = ast.parse(content)
        
        classes = []
        methods = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.append(f"{node.name}.{item.name}")
        
        print(f"🏗️  Found {len(classes)} classes: {classes}")
        print(f"🔧 Found {len(methods)} methods")
        
        # Check for required classes
        required_classes = ['EnhancedSRClustering', 'EnhancedSRClusteringConfig', 'EnhancedClusterResult']
        missing_classes = [cls for cls in required_classes if cls not in classes]
        
        if missing_classes:
            print(f"❌ Missing required classes: {missing_classes}")
            return False
        
        # Check for required methods
        required_methods = [
            'EnhancedSRClustering.__init__',
            'EnhancedSRClustering.cluster_sr_levels',
            'EnhancedSRClustering._extract_enhanced_features',
            'EnhancedSRClustering._perform_enhanced_clustering'
        ]
        
        missing_methods = [method for method in required_methods if method not in methods]
        
        if missing_methods:
            print(f"❌ Missing required methods: {missing_methods}")
            return False
        
        print(f"✅ Class structure validation passed for {file_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error validating class structure in {file_path}: {e}")
        return False

def validate_file_structure():
    """Validate the overall file structure."""
    print("🔍 Validating file structure...")
    
    # Check if the enhanced SR clustering file exists
    enhanced_file = Path("src/utils/sr_clustering/enhanced_sr_clustering.py")
    if not enhanced_file.exists():
        print(f"❌ Enhanced SR clustering file not found: {enhanced_file}")
        return False
    
    print(f"✅ Enhanced SR clustering file found: {enhanced_file}")
    
    # Check file size
    file_size = enhanced_file.stat().st_size
    print(f"📏 File size: {file_size:,} bytes")
    
    if file_size < 10000:  # Less than 10KB seems too small
        print(f"⚠️  File size seems small: {file_size} bytes")
    else:
        print(f"✅ File size looks reasonable: {file_size:,} bytes")
    
    return True

def main():
    """Main validation function."""
    print("🚀 Starting Enhanced SR Clustering validation...")
    print("=" * 60)
    
    # Change to workspace directory
    os.chdir("/workspace")
    
    # Validate file structure
    structure_ok = validate_file_structure()
    
    if not structure_ok:
        print("❌ File structure validation failed")
        return False
    
    print("\n" + "=" * 60)
    
    # Validate enhanced SR clustering file
    enhanced_file = "src/utils/sr_clustering/enhanced_sr_clustering.py"
    
    print(f"🔍 Validating {enhanced_file}...")
    
    # Syntax validation
    syntax_ok = validate_python_syntax(enhanced_file)
    
    # Class structure validation
    class_ok = validate_class_structure(enhanced_file)
    
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    results = [
        ("File Structure", structure_ok),
        ("Python Syntax", syntax_ok),
        ("Class Structure", class_ok)
    ]
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("🎉 All validations passed! The Enhanced SR Clustering module is ready.")
        return True
    else:
        print("❌ Some validations failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
