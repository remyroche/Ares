#!/usr/bin/env python3
"""
Simple validation script to check VectorBT optimization implementation in regime volatility features.
This script validates the code structure without requiring external dependencies.
"""

import os
import re
from typing import List, Dict, Any

def check_file_for_vectorbt_usage(file_path: str) -> Dict[str, Any]:
    """Check a file for VectorBT usage patterns."""
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    patterns = {
        'vectorbt_optimizer_init': r'vectorbt_optimizer\s*=\s*get_vectorbt_rolling_optimizer',
        'unified_optimizer_init': r'unified_optimizer\s*=\s*get_unified_optimization_system',
        'vectorbt_rolling_operation': r'_vectorbt_rolling_operation\s*\(',
        'vectorbt_optimizer_usage': r'self\.vectorbt_optimizer\.',
        'unified_optimizer_usage': r'self\.unified_optimizer\.',
        'rolling_apply_optimized': r'rolling_apply\s*\([^)]*func=',
        'direct_rolling_usage': r'\.rolling\s*\(',
        'vectorbt_imports': r'from.*vectorbt.*import|import.*vectorbt',
        'optimization_available': r'OPTIMIZATION_AVAILABLE',
        'fallback_handling': r'except.*Exception.*fallback|pandas.*fallback'
    }
    
    results = {}
    for pattern_name, pattern in patterns.items():
        matches = re.findall(pattern, content, re.MULTILINE)
        results[pattern_name] = {
            'count': len(matches),
            'matches': matches[:5] if matches else []  # Show first 5 matches
        }
    
    return results

def validate_regime_volatility_optimization():
    """Validate VectorBT optimization in regime volatility features."""
    print("🔍 Validating VectorBT optimization in regime volatility features...\n")
    
    files_to_check = [
        'src/feature_generation/categories/regime_volatility.py',
        'src/feature_generation/categories/regime_feature_integration.py',
        'src/feature_generation/categories/advanced_volatility_features.py'
    ]
    
    total_score = 0
    max_score = 0
    
    for file_path in files_to_check:
        print(f"📁 Checking {file_path}...")
        results = check_file_for_vectorbt_usage(file_path)
        
        if 'error' in results:
            print(f"❌ {results['error']}")
            continue
        
        file_score = 0
        file_max_score = 0
        
        # Check for proper VectorBT optimizer initialization
        if results['vectorbt_optimizer_init']['count'] > 0:
            print("✅ VectorBT optimizer initialization found")
            file_score += 2
        else:
            print("❌ VectorBT optimizer initialization missing")
        file_max_score += 2
        
        # Check for unified optimizer initialization
        if results['unified_optimizer_init']['count'] > 0:
            print("✅ Unified optimizer initialization found")
            file_score += 2
        else:
            print("❌ Unified optimizer initialization missing")
        file_max_score += 2
        
        # Check for VectorBT rolling operation usage
        if results['vectorbt_rolling_operation']['count'] > 0:
            print(f"✅ VectorBT rolling operations found ({results['vectorbt_rolling_operation']['count']} uses)")
            file_score += 3
        else:
            print("❌ VectorBT rolling operations not found")
        file_max_score += 3
        
        # Check for optimizer usage
        if results['vectorbt_optimizer_usage']['count'] > 0:
            print(f"✅ VectorBT optimizer usage found ({results['vectorbt_optimizer_usage']['count']} uses)")
            file_score += 2
        else:
            print("❌ VectorBT optimizer usage not found")
        file_max_score += 2
        
        # Check for unified optimizer usage
        if results['unified_optimizer_usage']['count'] > 0:
            print(f"✅ Unified optimizer usage found ({results['unified_optimizer_usage']['count']} uses)")
            file_score += 1
        else:
            print("⚠️ Unified optimizer usage not found")
        file_max_score += 1
        
        # Check for fallback handling
        if results['fallback_handling']['count'] > 0:
            print(f"✅ Fallback handling found ({results['fallback_handling']['count']} instances)")
            file_score += 2
        else:
            print("❌ Fallback handling not found")
        file_max_score += 2
        
        # Check for direct rolling usage (should be minimized)
        direct_rolling = results['direct_rolling_usage']['count']
        if direct_rolling == 0:
            print("✅ No direct pandas rolling usage found")
            file_score += 2
        elif direct_rolling < 5:
            print(f"⚠️ Limited direct pandas rolling usage ({direct_rolling} instances)")
            file_score += 1
        else:
            print(f"❌ Too much direct pandas rolling usage ({direct_rolling} instances)")
        file_max_score += 2
        
        print(f"📊 File score: {file_score}/{file_max_score} ({file_score/file_max_score*100:.1f}%)")
        print()
        
        total_score += file_score
        max_score += file_max_score
    
    return total_score, max_score

def check_optimization_consistency():
    """Check for consistency in VectorBT optimization usage."""
    print("🔍 Checking optimization consistency...\n")
    
    # Check if all files use similar patterns
    files = [
        'src/feature_generation/categories/regime_volatility.py',
        'src/feature_generation/categories/regime_feature_integration.py',
        'src/feature_generation/categories/advanced_volatility_features.py'
    ]
    
    patterns_found = set()
    
    for file_path in files:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
            
            if 'vectorbt_optimizer' in content:
                patterns_found.add('vectorbt_optimizer')
            if 'unified_optimizer' in content:
                patterns_found.add('unified_optimizer')
            if '_vectorbt_rolling_operation' in content:
                patterns_found.add('vectorbt_rolling_operation')
            if 'OPTIMIZATION_AVAILABLE' in content:
                patterns_found.add('optimization_available')
    
    print(f"✅ Consistent patterns found: {', '.join(patterns_found)}")
    
    expected_patterns = {'vectorbt_optimizer', 'vectorbt_rolling_operation', 'optimization_available'}
    missing_patterns = expected_patterns - patterns_found
    
    if missing_patterns:
        print(f"❌ Missing patterns: {', '.join(missing_patterns)}")
        return False
    else:
        print("✅ All expected patterns found consistently")
        return True

def main():
    """Run the validation."""
    print("🚀 Validating VectorBT optimization in regime volatility features...\n")
    
    # Validate optimization implementation
    total_score, max_score = validate_regime_volatility_optimization()
    
    # Check consistency
    consistency_ok = check_optimization_consistency()
    
    # Calculate overall score
    overall_score = total_score / max_score if max_score > 0 else 0
    consistency_bonus = 0.1 if consistency_ok else 0
    final_score = min(1.0, overall_score + consistency_bonus)
    
    print(f"\n📊 Overall Results:")
    print(f"   Implementation Score: {total_score}/{max_score} ({overall_score*100:.1f}%)")
    print(f"   Consistency: {'✅ Pass' if consistency_ok else '❌ Fail'}")
    print(f"   Final Score: {final_score*100:.1f}%")
    
    if final_score >= 0.8:
        print("\n🎉 VectorBT optimization implementation looks good!")
        return True
    elif final_score >= 0.6:
        print("\n⚠️ VectorBT optimization implementation needs improvement.")
        return False
    else:
        print("\n❌ VectorBT optimization implementation needs significant work.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)