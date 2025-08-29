#!/usr/bin/env python3
"""
Dependency-free validation script for advanced S/R methods in sr_breakout_predictor.py
"""

import ast
import sys
from pathlib import Path

def check_file_syntax(file_path: str) -> bool:
    """Check if a Python file has valid syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            ast.parse(f.read())
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in {file_path}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return False

def find_advanced_sr_methods(file_path: str) -> dict[str, list[int]]:
    """Find advanced S/R method definitions in the file."""
    advanced_methods = {
        'calculate_fibonacci_levels': [],
        'detect_elliott_wave_levels': [],
        'analyze_order_flow_levels': [],
        'detect_multi_timeframe_confluence': [],
        'get_comprehensive_sr_analysis': [],
        '_find_elliott_wave_points': [],
        '_calculate_volume_profile': [],
        '_detect_order_imbalances': []
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            
        for line_num, line in enumerate(lines, 1):
            for method in advanced_methods.keys():
                if f"async def {method}" in line or f"def {method}" in line:
                    advanced_methods[method].append(line_num)
                    
    except Exception as e:
        print(f"❌ Error analyzing {file_path}: {e}")
        
    return advanced_methods

def check_method_implementation(file_path: str, method_name: str) -> dict[str, bool]:
    """Check if a method is properly implemented."""
    implementation_checks = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for method definition
        if f"async def {method_name}" in content or f"def {method_name}" in content:
            implementation_checks[f"{method_name}_defined"] = True
        else:
            implementation_checks[f"{method_name}_defined"] = False
            
        # Check for docstring
        if f'"""{method_name}' in content or f"'{method_name}" in content:
            implementation_checks[f"{method_name}_documented"] = True
        else:
            implementation_checks[f"{method_name}_documented"] = False
            
        # Check for error handling
        if "try:" in content and "except Exception as e:" in content:
            implementation_checks[f"{method_name}_error_handling"] = True
        else:
            implementation_checks[f"{method_name}_error_handling"] = False
            
        # Check for logging
        if "self.logger" in content:
            implementation_checks[f"{method_name}_logging"] = True
        else:
            implementation_checks[f"{method_name}_logging"] = False
            
        # Check for validator decorator
        if "@validate_data_quality" in content:
            implementation_checks[f"{method_name}_validated"] = True
        else:
            implementation_checks[f"{method_name}_validated"] = False
            
    except Exception as e:
        print(f"❌ Error checking implementation for {method_name}: {e}")
        
    return implementation_checks

def check_integration_with_existing_methods(file_path: str) -> dict[str, bool]:
    """Check if advanced methods are integrated with existing methods."""
    integration_checks = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check if advanced methods are called in get_sr_context
        if "fibonacci_levels = await self.calculate_fibonacci_levels" in content:
            integration_checks["fibonacci_integrated"] = True
        else:
            integration_checks["fibonacci_integrated"] = False
            
        if "elliott_wave_levels = await self.detect_elliott_wave_levels" in content:
            integration_checks["elliott_integrated"] = True
        else:
            integration_checks["elliott_integrated"] = False
            
        if "order_flow_analysis = await self.analyze_order_flow_levels" in content:
            integration_checks["order_flow_integrated"] = True
        else:
            integration_checks["order_flow_integrated"] = False
            
        # Check if advanced methods are included in context
        if '"fibonacci_levels": fibonacci_levels' in content:
            integration_checks["fibonacci_in_context"] = True
        else:
            integration_checks["fibonacci_in_context"] = False
            
        if '"elliott_wave_levels": elliott_wave_levels' in content:
            integration_checks["elliott_in_context"] = True
        else:
            integration_checks["elliott_in_context"] = False
            
        if '"order_flow_analysis": order_flow_analysis' in content:
            integration_checks["order_flow_in_context"] = True
        else:
            integration_checks["order_flow_in_context"] = False
            
    except Exception as e:
        print(f"❌ Error checking integration: {e}")
        
    return integration_checks

def validate_advanced_sr_methods():
    """Validate advanced S/R methods implementation."""
    print("🚀 Validating Advanced S/R Methods Implementation")
    print("=" * 70)
    
    sr_file = "src/tactician/sr_breakout_predictor.py"
    
    if not Path(sr_file).exists():
        print(f"❌ S/R predictor file not found: {sr_file}")
        return False
        
    # Check syntax
    print("\n📋 Checking file syntax...")
    if not check_file_syntax(sr_file):
        print("❌ S/R predictor file has syntax errors")
        return False
    print("✅ File syntax is valid")
    
    # Find advanced methods
    print("\n🔍 Finding advanced S/R methods...")
    advanced_methods = find_advanced_sr_methods(sr_file)
    
    found_methods = []
    for method, lines in advanced_methods.items():
        if lines:
            found_methods.append(method)
            print(f"✅ {method}: Found at line(s) {lines}")
        else:
            print(f"❌ {method}: Not found")
    
    if not found_methods:
        print("❌ No advanced S/R methods found")
        return False
    
    # Check implementation quality
    print("\n🔧 Checking implementation quality...")
    implementation_results = {}
    for method in found_methods:
        checks = check_method_implementation(sr_file, method)
        implementation_results[method] = checks
        
        print(f"\n📊 {method}:")
        for check_name, result in checks.items():
            status = "✅" if result else "❌"
            print(f"   {status} {check_name}: {result}")
    
    # Check integration
    print("\n🔗 Checking integration with existing methods...")
    integration_checks = check_integration_with_existing_methods(sr_file)
    
    print("\n📊 Integration Status:")
    for check_name, result in integration_checks.items():
        status = "✅" if result else "❌"
        print(f"   {status} {check_name}: {result}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 ADVANCED S/R METHODS VALIDATION SUMMARY")
    print("=" * 70)
    
    total_methods = len(advanced_methods)
    found_count = len(found_methods)
    
    print(f"Total Advanced Methods: {total_methods}")
    print(f"Methods Found: {found_count}")
    print(f"Methods Missing: {total_methods - found_count}")
    print(f"Success Rate: {found_count/total_methods*100:.1f}%")
    
    # Implementation quality summary
    total_checks = 0
    passed_checks = 0
    
    for method, checks in implementation_results.items():
        for check_name, result in checks.items():
            total_checks += 1
            if result:
                passed_checks += 1
    
    if total_checks > 0:
        print(f"\nImplementation Quality:")
        print(f"Total Checks: {total_checks}")
        print(f"Passed Checks: {passed_checks}")
        print(f"Quality Score: {passed_checks/total_checks*100:.1f}%")
    
    # Integration summary
    integration_passed = sum(integration_checks.values())
    integration_total = len(integration_checks)
    
    print(f"\nIntegration Status:")
    print(f"Integration Checks: {integration_passed}/{integration_total}")
    print(f"Integration Score: {integration_passed/integration_total*100:.1f}%")
    
    # Overall result
    if found_count == total_methods and integration_passed == integration_total:
        print("\n🎉 ALL ADVANCED S/R METHODS VALIDATIONS PASSED!")
        print("The advanced S/R methods are properly implemented and integrated.")
        return True
    else:
        print(f"\n⚠️ {total_methods - found_count} METHODS MISSING OR {integration_total - integration_passed} INTEGRATION ISSUES")
        print("Some advanced S/R methods need attention.")
        return False

def analyze_method_details():
    """Analyze specific details of the advanced methods."""
    print("\n🔍 Advanced Methods Analysis")
    print("=" * 50)
    
    sr_file = "src/tactician/sr_breakout_predictor.py"
    
    try:
        with open(sr_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for specific features
        features = {
            "Fibonacci Levels": "calculate_fibonacci_levels" in content,
            "Elliott Wave Analysis": "detect_elliott_wave_levels" in content,
            "Order Flow Analysis": "analyze_order_flow_levels" in content,
            "POC Detection": "poc_level" in content,
            "HVN Detection": "hvn_levels" in content,
            "Volume Profile": "volume_profile" in content,
            "Multi-Timeframe Confluence": "detect_multi_timeframe_confluence" in content,
            "Comprehensive Analysis": "get_comprehensive_sr_analysis" in content
        }
        
        print("\n📋 Feature Analysis:")
        for feature, implemented in features.items():
            status = "✅" if implemented else "❌"
            print(f"   {status} {feature}: {'Implemented' if implemented else 'Not Found'}")
            
    except Exception as e:
        print(f"❌ Error analyzing method details: {e}")

if __name__ == "__main__":
    # Validate advanced methods
    validation_passed = validate_advanced_sr_methods()
    
    # Analyze method details
    analyze_method_details()
    
    # Overall result
    if validation_passed:
        print("\n🎉 ADVANCED S/R METHODS IMPLEMENTATION SUCCESSFUL!")
        sys.exit(0)
    else:
        print("\n⚠️ ADVANCED S/R METHODS NEED ATTENTION!")
        sys.exit(1)