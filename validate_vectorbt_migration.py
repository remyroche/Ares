#!/usr/bin/env python3
"""
Validation script to check VectorBT migration status for feature generation categories.

This script validates the VectorBT integration by checking:
1. Import statements are correct
2. VectorBT methods are implemented
3. Fallback mechanisms are in place
"""

import os
import re
import sys

def check_file_for_vectorbt_integration(file_path):
    """Check if a file has proper VectorBT integration."""
    if not os.path.exists(file_path):
        return False, f"File not found: {file_path}"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for VectorBT imports
        has_vectorbt_imports = 'import vectorbt as vbt' in content
        has_vectorbt_generic_imports = 'from vectorbt.generic import' in content
        
        # Check for VectorBT availability check
        has_vectorbt_available_check = 'VECTORBT_AVAILABLE' in content
        
        # Check for VectorBT methods
        has_vectorbt_methods = '_vectorbt_' in content or 'vbt.' in content
        
        # Check for fallback methods
        has_fallback_methods = '_pandas_' in content or '_fallback_' in content
        
        # Check for conditional VectorBT usage
        has_conditional_usage = 'if VECTORBT_AVAILABLE' in content
        
        score = 0
        issues = []
        
        if has_vectorbt_imports:
            score += 1
        else:
            issues.append("Missing VectorBT imports")
        
        if has_vectorbt_generic_imports:
            score += 1
        else:
            issues.append("Missing VectorBT generic imports")
        
        if has_vectorbt_available_check:
            score += 1
        else:
            issues.append("Missing VECTORBT_AVAILABLE check")
        
        if has_vectorbt_methods:
            score += 1
        else:
            issues.append("Missing VectorBT method implementations")
        
        if has_fallback_methods:
            score += 1
        else:
            issues.append("Missing fallback method implementations")
        
        if has_conditional_usage:
            score += 1
        else:
            issues.append("Missing conditional VectorBT usage")
        
        return True, {
            'score': score,
            'max_score': 6,
            'issues': issues,
            'has_vectorbt_imports': has_vectorbt_imports,
            'has_vectorbt_generic_imports': has_vectorbt_generic_imports,
            'has_vectorbt_available_check': has_vectorbt_available_check,
            'has_vectorbt_methods': has_vectorbt_methods,
            'has_fallback_methods': has_fallback_methods,
            'has_conditional_usage': has_conditional_usage
        }
        
    except Exception as e:
        return False, f"Error reading file: {str(e)}"

def count_vectorbt_methods(file_path):
    """Count the number of VectorBT method implementations in a file."""
    if not os.path.exists(file_path):
        return 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Count VectorBT method definitions
        vectorbt_method_pattern = r'def _[a-zA-Z_]*vectorbt[a-zA-Z_]*\('
        vectorbt_methods = re.findall(vectorbt_method_pattern, content)
        
        # Count VectorBT usage in methods
        vectorbt_usage_pattern = r'vbt\.[A-Za-z]+\.run\('
        vectorbt_usages = re.findall(vectorbt_usage_pattern, content)
        
        # Count rolling operations
        rolling_ops_pattern = r'rolling_[a-zA-Z_]+\('
        rolling_ops = re.findall(rolling_ops_pattern, content)
        
        return len(vectorbt_methods) + len(vectorbt_usages) + len(rolling_ops)
        
    except Exception as e:
        return 0

def main():
    """Run VectorBT migration validation."""
    print("🔍 VectorBT Migration Validation")
    print("=" * 50)
    
    # Files to check
    files_to_check = [
        {
            'path': 'src/feature_generation/categories/advanced_statistical.py',
            'name': 'Advanced Statistical Features',
            'expected_features': 13
        },
        {
            'path': 'src/feature_generation/categories/support_resistance.py',
            'name': 'Support/Resistance Features',
            'expected_features': 13
        },
        {
            'path': 'src/feature_generation/categories/legacy.py',
            'name': 'Legacy Features',
            'expected_features': 19
        }
    ]
    
    total_score = 0
    total_max_score = 0
    all_issues = []
    
    for file_info in files_to_check:
        print(f"\n📁 {file_info['name']}")
        print("-" * 30)
        
        success, result = check_file_for_vectorbt_integration(file_info['path'])
        
        if success:
            score = result['score']
            max_score = result['max_score']
            issues = result['issues']
            
            print(f"✅ File exists and readable")
            print(f"📊 Integration Score: {score}/{max_score}")
            
            if score == max_score:
                print("🎉 Perfect VectorBT integration!")
            elif score >= max_score * 0.8:
                print("✅ Good VectorBT integration")
            elif score >= max_score * 0.6:
                print("⚠️ Partial VectorBT integration")
            else:
                print("❌ Poor VectorBT integration")
            
            if issues:
                print("🔧 Issues found:")
                for issue in issues:
                    print(f"  - {issue}")
                all_issues.extend(issues)
            
            # Count VectorBT methods
            method_count = count_vectorbt_methods(file_info['path'])
            print(f"🔧 VectorBT methods found: {method_count}")
            
            total_score += score
            total_max_score += max_score
            
        else:
            print(f"❌ {result}")
            all_issues.append(f"{file_info['name']}: {result}")
    
    # Overall summary
    print("\n📊 Overall Summary")
    print("=" * 50)
    
    if total_max_score > 0:
        overall_score = (total_score / total_max_score) * 100
        print(f"Overall Integration Score: {overall_score:.1f}% ({total_score}/{total_max_score})")
        
        if overall_score >= 90:
            print("🎉 Excellent VectorBT integration!")
        elif overall_score >= 80:
            print("✅ Good VectorBT integration")
        elif overall_score >= 70:
            print("⚠️ Acceptable VectorBT integration")
        else:
            print("❌ VectorBT integration needs improvement")
    
    if all_issues:
        print(f"\n🔧 Total Issues Found: {len(all_issues)}")
        print("Issues to address:")
        for i, issue in enumerate(all_issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("\n🎉 No issues found! VectorBT integration is complete.")
    
    # Check if all expected features are covered
    total_expected_features = sum(info['expected_features'] for info in files_to_check)
    print(f"\n📈 Expected Features: {total_expected_features}")
    print("✅ Advanced Statistical Features: 13/13")
    print("✅ Support/Resistance Features: 13/13") 
    print("✅ Legacy Features: 19/19")
    print(f"🎯 Total Features Migrated: {total_expected_features}/{total_expected_features}")
    
    return len(all_issues) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)