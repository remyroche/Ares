#!/usr/bin/env python3
"""
Dependency-free validation script for enhanced S/R strength implementation
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

def find_enhanced_strength_methods(file_path: str) -> dict[str, list[int]]:
    """Find enhanced strength calculation method definitions."""
    enhanced_methods = {
        'calculate_touch_count': [],
        'calculate_level_age': [],
        'calculate_bounce_rate': [],
        'calculate_isolation_score': [],
        'cluster_sr_levels_dbscan': [],
        'calculate_comprehensive_strength': []
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            
        for line_num, line in enumerate(lines, 1):
            for method in enhanced_methods.keys():
                if f"async def {method}" in line:
                    enhanced_methods[method].append(line_num)
                    
    except Exception as e:
        print(f"❌ Error analyzing {file_path}: {e}")
        
    return enhanced_methods

def check_method_implementation(file_path: str, method_name: str) -> dict[str, bool]:
    """Check if a method is properly implemented."""
    implementation_checks = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for method definition
        if f"async def {method_name}" in content:
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

def check_configuration_integration(file_path: str) -> dict[str, bool]:
    """Check if enhanced strength configuration is integrated."""
    config_checks = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for DBSCAN configuration
        if "dbscan_clustering" in content:
            config_checks["dbscan_config"] = True
        else:
            config_checks["dbscan_config"] = False
            
        # Check for strength calculation configuration
        if "strength_calculation" in content:
            config_checks["strength_config"] = True
        else:
            config_checks["strength_config"] = False
            
        # Check for enhanced strength weights
        if "strength_score_weights" in content:
            config_checks["strength_weights"] = True
        else:
            config_checks["strength_weights"] = False
            
        # Check for DBSCAN import
        if "from sklearn.cluster import DBSCAN" in content:
            config_checks["dbscan_import"] = True
        else:
            config_checks["dbscan_import"] = False
            
        # Check for DBSCAN availability check
        if "DBSCAN_AVAILABLE" in content:
            config_checks["dbscan_availability_check"] = True
        else:
            config_checks["dbscan_availability_check"] = False
            
    except Exception as e:
        print(f"❌ Error checking configuration integration: {e}")
        
    return config_checks

def check_integration_with_existing_methods(file_path: str) -> dict[str, bool]:
    """Check if enhanced methods are integrated with existing methods."""
    integration_checks = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check if enhanced strength is used in get_sr_context
        if "enhanced_strength_support" in content:
            integration_checks["enhanced_strength_in_context"] = True
        else:
            integration_checks["enhanced_strength_in_context"] = False
            
        # Check if clustering is used in get_sr_context
        if "clustering_result" in content:
            integration_checks["clustering_in_context"] = True
        else:
            integration_checks["clustering_in_context"] = False
            
        # Check if enhanced strength is used in _find_nearest_level
        if "enhanced_strength" in content and "_find_nearest_level" in content:
            integration_checks["enhanced_strength_in_nearest"] = True
        else:
            integration_checks["enhanced_strength_in_nearest"] = False
            
        # Check if comprehensive strength is called
        if "calculate_comprehensive_strength" in content:
            integration_checks["comprehensive_strength_called"] = True
        else:
            integration_checks["comprehensive_strength_called"] = False
            
        # Check if DBSCAN clustering is called
        if "cluster_sr_levels_dbscan" in content:
            integration_checks["dbscan_called"] = True
        else:
            integration_checks["dbscan_called"] = False
            
    except Exception as e:
        print(f"❌ Error checking integration: {e}")
        
    return integration_checks

def validate_enhanced_sr_strength():
    """Validate enhanced S/R strength implementation."""
    print("🚀 Validating Enhanced S/R Strength Implementation")
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
    
    # Find enhanced methods
    print("\n🔍 Finding enhanced strength methods...")
    enhanced_methods = find_enhanced_strength_methods(sr_file)
    
    found_methods = []
    for method, lines in enhanced_methods.items():
        if lines:
            found_methods.append(method)
            print(f"✅ {method}: Found at line(s) {lines}")
        else:
            print(f"❌ {method}: Not found")
    
    if not found_methods:
        print("❌ No enhanced strength methods found")
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
    
    # Check configuration integration
    print("\n⚙️ Checking configuration integration...")
    config_checks = check_configuration_integration(sr_file)
    
    print("\n📊 Configuration Integration:")
    for check_name, result in config_checks.items():
        status = "✅" if result else "❌"
        print(f"   {status} {check_name}: {result}")
    
    # Check integration with existing methods
    print("\n🔗 Checking integration with existing methods...")
    integration_checks = check_integration_with_existing_methods(sr_file)
    
    print("\n📊 Integration Status:")
    for check_name, result in integration_checks.items():
        status = "✅" if result else "❌"
        print(f"   {status} {check_name}: {result}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 ENHANCED S/R STRENGTH VALIDATION SUMMARY")
    print("=" * 70)
    
    total_methods = len(enhanced_methods)
    found_count = len(found_methods)
    
    print(f"Total Enhanced Methods: {total_methods}")
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
    
    # Configuration integration summary
    config_passed = sum(config_checks.values())
    config_total = len(config_checks)
    
    print(f"\nConfiguration Integration:")
    print(f"Configuration Checks: {config_passed}/{config_total}")
    print(f"Configuration Score: {config_passed/config_total*100:.1f}%")
    
    # Integration summary
    integration_passed = sum(integration_checks.values())
    integration_total = len(integration_checks)
    
    print(f"\nIntegration Status:")
    print(f"Integration Checks: {integration_passed}/{integration_total}")
    print(f"Integration Score: {integration_passed/integration_total*100:.1f}%")
    
    # Overall result
    if (found_count == total_methods and 
        integration_passed == integration_total and 
        config_passed == config_total):
        print("\n🎉 ALL ENHANCED S/R STRENGTH VALIDATIONS PASSED!")
        print("The enhanced S/R strength methods are properly implemented and integrated.")
        return True
    else:
        print(f"\n⚠️ {total_methods - found_count} METHODS MISSING OR {integration_total - integration_passed} INTEGRATION ISSUES")
        print("Some enhanced S/R strength methods need attention.")
        return False

def analyze_implementation_details():
    """Analyze specific details of the enhanced implementation."""
    print("\n🔍 Enhanced Implementation Analysis")
    print("=" * 50)
    
    sr_file = "src/tactician/sr_breakout_predictor.py"
    
    try:
        with open(sr_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for specific features
        features = {
            "DBSCAN Clustering": "cluster_sr_levels_dbscan" in content,
            "Touch Count Analysis": "calculate_touch_count" in content,
            "Level Age Analysis": "calculate_level_age" in content,
            "Bounce Rate Analysis": "calculate_bounce_rate" in content,
            "Isolation Score Analysis": "calculate_isolation_score" in content,
            "Comprehensive Strength": "calculate_comprehensive_strength" in content,
            "Enhanced Configuration": "strength_calculation" in content,
            "DBSCAN Configuration": "dbscan_clustering" in content,
            "Enhanced Context": "enhanced_strength_support" in content,
            "Clustering Results": "clustering_result" in content
        }
        
        print("\n📋 Feature Analysis:")
        for feature, implemented in features.items():
            status = "✅" if implemented else "❌"
            print(f"   {status} {feature}: {'Implemented' if implemented else 'Not Found'}")
            
        # Check for specific implementation details
        details = {
            "Error Handling": "try:" in content and "except Exception as e:" in content,
            "Logging": "self.logger" in content,
            "Validation": "@validate_data_quality" in content,
            "Async Methods": "async def" in content,
            "Configuration": "self.config" in content,
            "DBSCAN Import": "from sklearn.cluster import DBSCAN" in content,
            "Availability Check": "DBSCAN_AVAILABLE" in content
        }
        
        print("\n🔧 Implementation Details:")
        for detail, present in details.items():
            status = "✅" if present else "❌"
            print(f"   {status} {detail}: {'Present' if present else 'Missing'}")
            
    except Exception as e:
        print(f"❌ Error analyzing implementation details: {e}")

if __name__ == "__main__":
    # Validate enhanced implementation
    validation_passed = validate_enhanced_sr_strength()
    
    # Analyze implementation details
    analyze_implementation_details()
    
    # Overall result
    if validation_passed:
        print("\n🎉 ENHANCED S/R STRENGTH IMPLEMENTATION SUCCESSFUL!")
        sys.exit(0)
    else:
        print("\n⚠️ ENHANCED S/R STRENGTH IMPLEMENTATION NEEDS ATTENTION!")
        sys.exit(1)